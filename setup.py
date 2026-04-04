from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "aleff1._aleff",
            sources=["src/aleff1/_aleff.c"],
        ),
    ],
)
