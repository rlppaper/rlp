from setuptools import Extension, setup

setup(
    name="rlp",
    description="Package containing a Gymnasium-compatible version of Simon Tatham's Portable Puzzle Collection.",
    version="1.0.0",
    install_requires=["gymnasium>=0.28.0", "pygame>=2.1.0"],
    package_dir={
        "": "rlp"
    },
    ext_modules=[Extension("constants", ["rlp/constants/puzzle_constants.c"])]
)