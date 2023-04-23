import os
from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# Package meta-data.
NAME = "learnware"
DESCRIPTION = "learnware market project"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = get_version("learnware/__init__.py")


# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")


MACOS = 0
WINDOWS = 1
LINUX = 2


def get_platform():
    import platform

    sys_platform = platform.platform().lower()
    if "windows" in sys_platform:
        return WINDOWS
    elif "macos" in sys_platform:
        return MACOS
    elif "linux" in sys_platform:
        return LINUX
    raise SystemError("Learnware only support MACOS/Linux/Windows")


# What packages are required for this module to be executed?
# `estimator` may depend on other packages. In order to reduce dependencies, it is not written here.
REQUIRED = [
    "numpy>=1.19.2",
    "pandas>=0.25.1",
    "scipy>=1.0.0",
    "matplotlib>=3.1.3",
    "torch>=1.11.0",
    "cvxopt>=1.3.0",
    "tqdm>=4.65.0",
    "scikit-learn>=0.22",
    "joblib>=1.2.0",
    "pyyaml>=6.0",
    "fire>=0.3.1",
    "lightgbm>=3.3.0",
    "psutil>=5.9.4",
    "torchvision>=0.15.1",
    "tensorflow>=2.4.0",
]

if get_platform() != MACOS:
    REQUIRED.append("faiss-cpu>=1.7.1")

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        license="MIT Licence",
        url="https://git.nju.edu.cn/learnware/learnware-market",
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
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
    )
