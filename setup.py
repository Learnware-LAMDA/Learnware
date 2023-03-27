import os
from setuptools import find_packages, setup

# Package meta-data.
NAME = "learnware"
DESCRIPTION = "learnware market project"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"


# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")

# What packages are required for this module to be executed?
# `estimator` may depend on other packages. In order to reduce dependencies, it is not written here.
REQUIRED = [
    "numpy>=1.16.0,<1.20.0",
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


if __name__ == "__main__":

    setup(
        name=NAME,
        version=VERSION,
        url="https://git.nju.edu.cn/",
        packages=find_packages(),
        include_package_data=True,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=REQUIRES_PYTHON,
        install_requires=REQUIRED,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
    )
