from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "aleff._aleff",
            sources=["src/aleff/_aleff.c"],
        ),
    ],
)
