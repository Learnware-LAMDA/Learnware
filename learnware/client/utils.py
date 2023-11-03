import os
import zipfile
import tempfile
import subprocess

from ..logger import get_module_logger
from .package_utils import filter_nonexist_conda_packages_file, filter_nonexist_pip_packages_file

logger = get_module_logger(module_name="client_utils")


def system_execute(args, timeout=None):
    com_process = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    try:
        com_process.check_returncode()
    except subprocess.CalledProcessError as err:
        logger.error(f"System Execute Error: {com_process.stderr.decode()}")
        raise err


def remove_enviroment(conda_env):
    system_execute(args=["conda", "env", "remove", "-n", f"{conda_env}"])


def install_environment(zip_path, conda_env):
    """Install environment of a learnware

    Parameters
    ----------
    zip_path : str
        Path of the learnware zip file
    conda_env : str
        a new conda environment will be created with the given name;

    Raises
    ------
    Exception
        Lack of the environment configuration file.
    """
    with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
        with zipfile.ZipFile(file=zip_path, mode="r") as z_file:
            logger.info(f"zip_file namelist: {z_file.namelist()}")
            if "environment.yaml" in z_file.namelist():
                z_file.extract(member="environment.yaml", path=tempdir)
                yaml_path: str = os.path.join(tempdir, "environment.yaml")
                yaml_path_filter: str = os.path.join(tempdir, "environment_filter.yaml")
                logger.info(f"checking the avaliabe conda packages for {conda_env}")
                filter_nonexist_conda_packages_file(yaml_file=yaml_path, output_yaml_file=yaml_path_filter)
                # create environment
                logger.info(f"create conda env [{conda_env}] according to .yaml file")
                system_execute(
                    args=["conda", "env", "create", "--name", f"{conda_env}", "--file", f"{yaml_path_filter}"]
                )

            elif "requirements.txt" in z_file.namelist():
                z_file.extract(member="requirements.txt", path=tempdir)
                requirements_path: str = os.path.join(tempdir, "requirements.txt")
                requirements_path_filter: str = os.path.join(tempdir, "requirements_filter.txt")
                logger.info(f"checking the avaliabe pip packages for {conda_env}")
                filter_nonexist_pip_packages_file(
                    requirements_file=requirements_path, output_file=requirements_path_filter
                )
                logger.info(f"create empty conda env [{conda_env}]")
                system_execute(args=["conda", "create", "-y", "--name", f"{conda_env}", "python=3.8"])
                logger.info(f"install pip requirements for conda env [{conda_env}]")
                system_execute(
                    args=[
                        "conda",
                        "run",
                        "-n",
                        f"{conda_env}",
                        "--no-capture-output",
                        "python3",
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        f"{requirements_path_filter}",
                        "--no-dependencies",
                    ]
                )
            else:
                raise Exception("Environment.yaml or requirements.txt not found in the learnware zip file.")

    logger.info(f"install learnware package for conda env [{conda_env}]")
    system_execute(
        args=[
            "conda",
            "run",
            "-n",
            f"{conda_env}",
            "--no-capture-output",
            "python3",
            "-m",
            "pip",
            "install",
            "learnware",
        ]
    )
