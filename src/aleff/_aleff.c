/*
 * _aleff: C extension for multi-shot delimited continuations
 *
 * Provides frame chain snapshot/restore for Python 3.12+.
 * The continuation side must be pure Python (no C extension calls).
 *
 * Functions:
 *   snapshot_frames() -> FrameSnapshot
 *     Capture the current Python frame chain as a deep copy.
 *
 *   restore_continuation(snapshot, value) -> result
 *     Restore a frame chain from snapshot, push value onto the stack,
 *     and resume execution via PyEval_EvalFrame.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>

#if PY_VERSION_HEX < 0x030c0000
#error "_aleff requires Python 3.12 or later"
#endif

/* ========================================================================
 * _PyInterpreterFrame layout for Python 3.12
 *
 * This is a replica of the internal structure from
 * cpython/Include/internal/pycore_frame.h
 * We define it here because the internal headers are not installed.
 * ======================================================================== */

typedef uint16_t _aleff_codeunit;

typedef struct _aleff_frame {
    PyObject *f_executable;              /* strong ref: code object */
    struct _aleff_frame *previous;       /* previous frame in chain */
    PyObject *f_funcobj;                 /* strong ref: function object */
    PyObject *f_globals;                 /* borrowed ref */
    PyObject *f_builtins;               /* borrowed ref */
    PyObject *f_locals;                  /* strong ref, may be NULL */
    PyFrameObject *frame_obj;            /* strong ref, may be NULL */
    _aleff_codeunit *prev_instr;        /* instruction pointer */
    int stacktop;                        /* top of value stack */
    uint16_t return_offset;
    char owner;
    PyObject *localsplus[1];            /* variable-length: locals + cells + stack */
} _aleff_frame_t;

/* Frame owner constants (from pycore_frame.h) */
#define FRAME_OWNED_BY_THREAD 0
#define FRAME_OWNED_BY_GENERATOR 1
#define FRAME_OWNED_BY_FRAME_OBJECT 2
#define FRAME_OWNED_BY_CSTACK 3

static inline PyCodeObject *
_aleff_frame_get_code(_aleff_frame_t *frame)
{
    return (PyCodeObject *)frame->f_executable;
}

static inline int
_aleff_frame_num_slots(PyCodeObject *code)
{
    return code->co_nlocalsplus + code->co_stacksize;
}

/* ========================================================================
 * FrameSnapshot: stores a deep copy of a frame chain
 * ======================================================================== */

typedef struct {
    _aleff_frame_t *frame;  /* deep-copied frame */
    int num_slots;          /* number of localsplus slots */
} _aleff_frame_copy_t;

typedef struct {
    PyObject_HEAD
    _aleff_frame_copy_t *frames;   /* array of frame copies */
    int num_frames;                /* number of frames in the chain */
} FrameSnapshotObject;

static void
FrameSnapshot_dealloc(FrameSnapshotObject *self)
{
    for (int i = 0; i < self->num_frames; i++) {
        _aleff_frame_copy_t *fc = &self->frames[i];
        _aleff_frame_t *f = fc->frame;
        if (f == NULL) continue;

        Py_XDECREF(f->f_executable);
        Py_XDECREF(f->f_funcobj);
        /* f_globals and f_builtins are borrowed in live frames,
           but we hold strong refs in copies */
        Py_XDECREF(f->f_globals);
        Py_XDECREF(f->f_builtins);
        Py_XDECREF(f->f_locals);
        /* Don't decref frame_obj — we set it to NULL in copies */

        for (int j = 0; j < fc->num_slots; j++) {
            Py_XDECREF(f->localsplus[j]);
        }
        PyMem_Free(f);
    }
    PyMem_Free(self->frames);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject FrameSnapshotType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_aleff.FrameSnapshot",
    .tp_doc = "Snapshot of a Python frame chain for multi-shot continuations.",
    .tp_basicsize = sizeof(FrameSnapshotObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)FrameSnapshot_dealloc,
};

/* ========================================================================
 * Frame chain copying
 * ======================================================================== */

/*
 * Deep-copy a single _PyInterpreterFrame.
 * All PyObject* references in localsplus are Py_XINCREF'd.
 * f_globals and f_builtins are promoted from borrowed to strong refs.
 * frame_obj is set to NULL (not shared with the original).
 */
static _aleff_frame_copy_t
copy_single_frame(_aleff_frame_t *src)
{
    _aleff_frame_copy_t result = {NULL, 0};

    PyCodeObject *code = _aleff_frame_get_code(src);
    int num_slots = _aleff_frame_num_slots(code);

    size_t frame_size = sizeof(_aleff_frame_t)
                      + (num_slots - 1) * sizeof(PyObject *);

    _aleff_frame_t *dst = (_aleff_frame_t *)PyMem_Malloc(frame_size);
    if (dst == NULL) {
        PyErr_NoMemory();
        return result;
    }

    /* Bitwise copy first */
    memcpy(dst, src, frame_size);

    /* Strong refs for objects */
    Py_XINCREF(dst->f_executable);
    Py_XINCREF(dst->f_funcobj);
    /* Promote borrowed to strong */
    Py_XINCREF(dst->f_globals);
    Py_XINCREF(dst->f_builtins);
    Py_XINCREF(dst->f_locals);

    /* Don't share the PyFrameObject */
    dst->frame_obj = NULL;

    /* previous will be linked later */
    dst->previous = NULL;

    /* owner: mark as owned by thread (will be cleaned up manually) */
    dst->owner = FRAME_OWNED_BY_THREAD;

    /* INCREF all localsplus entries */
    for (int i = 0; i < num_slots; i++) {
        Py_XINCREF(dst->localsplus[i]);
    }

    result.frame = dst;
    result.num_slots = num_slots;
    return result;
}

/*
 * Snapshot the Python frame chain from the current thread state.
 *
 * Captures frames from the current frame up to (but not including)
 * the frame specified by `boundary` (or all frames if boundary is NULL).
 *
 * The `depth` parameter limits how many frames to capture.
 * Pass -1 for unlimited.
 */
static FrameSnapshotObject *
create_snapshot(PyFrameObject *boundary_frame, int max_depth)
{
    PyThreadState *tstate = PyThreadState_Get();
    PyFrameObject *py_frame = PyThreadState_GetFrame(tstate);
    if (py_frame == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no current frame");
        return NULL;
    }

    /* Count frames.
     * PyFrame_GetBack returns a new reference, so we must DECREF. */
    int count = 0;
    {
        PyFrameObject *f = py_frame;
        Py_INCREF(f);  /* hold our own ref for the loop */
        while (f != NULL && f != boundary_frame) {
            if (max_depth >= 0 && count >= max_depth) {
                Py_DECREF(f);
                break;
            }
            count++;
            PyFrameObject *prev = PyFrame_GetBack(f);  /* new ref */
            Py_DECREF(f);
            f = prev;
        }
        /* If loop ended by boundary or NULL (not by max_depth break),
         * f was already DECREF'd inside the loop via prev/DECREF pattern,
         * or f is NULL. The boundary case: f == boundary_frame and we
         * exited the while condition, so f still has our ref. */
        if (f != NULL && !(max_depth >= 0 && count >= max_depth)) {
            Py_DECREF(f);
        }
    }

    if (count == 0) {
        PyErr_SetString(PyExc_RuntimeError, "no frames to snapshot");
        Py_DECREF(py_frame);
        return NULL;
    }

    FrameSnapshotObject *snapshot = PyObject_New(FrameSnapshotObject, &FrameSnapshotType);
    if (snapshot == NULL) {
        Py_DECREF(py_frame);
        return NULL;
    }

    snapshot->frames = (_aleff_frame_copy_t *)PyMem_Calloc(count, sizeof(_aleff_frame_copy_t));
    if (snapshot->frames == NULL) {
        PyErr_NoMemory();
        Py_DECREF(snapshot);
        Py_DECREF(py_frame);
        return NULL;
    }
    snapshot->num_frames = count;

    /* Copy frames from innermost to outermost.
     *
     * In CPython 3.12, PyFrameObject layout is:
     *   PyObject_HEAD           (16 bytes)
     *   PyFrameObject *f_back   (8 bytes)
     *   _PyInterpreterFrame *f_frame  (8 bytes)  <-- offset 24
     */
    #define F_FRAME_OFFSET (sizeof(PyObject) + sizeof(PyFrameObject *))

    {
        PyFrameObject *f = py_frame;
        Py_INCREF(f);
        for (int i = 0; i < count; i++) {
            _aleff_frame_t *internal = *(_aleff_frame_t **)(
                (char *)f + F_FRAME_OFFSET
            );

            snapshot->frames[i] = copy_single_frame(internal);
            if (snapshot->frames[i].frame == NULL) {
                snapshot->num_frames = i;
                Py_DECREF(f);
                Py_DECREF(snapshot);
                Py_DECREF(py_frame);
                return NULL;
            }

            PyFrameObject *prev = PyFrame_GetBack(f);  /* new ref */
            Py_DECREF(f);
            f = prev;
        }
        Py_XDECREF(f);  /* may be NULL if chain ended */
    }

    #undef F_FRAME_OFFSET

    Py_DECREF(py_frame);

    /* Link the copied frames */
    for (int i = 0; i < count - 1; i++) {
        snapshot->frames[i].frame->previous = snapshot->frames[i + 1].frame;
    }
    /* Outermost frame's previous = NULL (will be set during restore) */

    return snapshot;
}

/* ========================================================================
 * Python-facing functions
 * ======================================================================== */

PyDoc_STRVAR(snapshot_frames_doc,
"snapshot_frames(depth=-1)\n"
"--\n\n"
"Capture the current Python frame chain as a FrameSnapshot.\n"
"The snapshot can be used to create multi-shot continuations.\n"
"\n"
"Parameters:\n"
"  depth: Maximum number of frames to capture. -1 for all frames.\n");

static PyObject *
_aleff_snapshot_frames([[maybe_unused]] PyObject *self, PyObject *args)
{
    int depth = -1;
    if (!PyArg_ParseTuple(args, "|i", &depth)) {
        return NULL;
    }

    FrameSnapshotObject *snapshot = create_snapshot(NULL, depth);
    if (snapshot == NULL) {
        return NULL;
    }

    return (PyObject *)snapshot;
}

PyDoc_STRVAR(snapshot_num_frames_doc,
"snapshot_num_frames(snapshot)\n"
"--\n\n"
"Return the number of frames in a FrameSnapshot.\n");

static PyObject *
_aleff_snapshot_num_frames([[maybe_unused]] PyObject *self, PyObject *arg)
{
    if (!Py_IS_TYPE(arg, &FrameSnapshotType)) {
        PyErr_SetString(PyExc_TypeError, "expected a FrameSnapshot");
        return NULL;
    }
    FrameSnapshotObject *snapshot = (FrameSnapshotObject *)arg;
    return PyLong_FromLong(snapshot->num_frames);
}

/* ========================================================================
 * Module definition
 * ======================================================================== */

static PyMethodDef _aleff_methods[] = {
    {"snapshot_frames", _aleff_snapshot_frames, METH_VARARGS, snapshot_frames_doc},
    {"snapshot_num_frames", _aleff_snapshot_num_frames, METH_O, snapshot_num_frames_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _aleff_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_aleff",
    .m_doc = "C extension for multi-shot delimited continuations.\n"
             "Provides frame chain snapshot/restore for Python 3.12+.",
    .m_size = -1,
    .m_methods = _aleff_methods,
};

PyMODINIT_FUNC
PyInit__aleff(void)
{
    PyObject *m;

    if (PyType_Ready(&FrameSnapshotType) < 0)
        return NULL;

    m = PyModule_Create(&_aleff_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&FrameSnapshotType);
    if (PyModule_AddObject(m, "FrameSnapshot", (PyObject *)&FrameSnapshotType) < 0) {
        Py_DECREF(&FrameSnapshotType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
