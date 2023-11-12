import os
import zipfile
import tempfile
import subprocess

from ..logger import get_module_logger
from .package_utils import filter_nonexist_conda_packages_file, filter_nonexist_pip_packages_file

logger = get_module_logger(module_name="client_utils")


def system_execute(args, timeout=None, env=None, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE):
    if env is None:
        env = os.environ.copy()
        pass

    if isinstance(args, str):
        pass
    else:
        args = " ".join(args)
        pass

    com_process = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=timeout, env=env, shell=True)

    try:
        com_process.check_returncode()
    except subprocess.CalledProcessError as err:
        errmsg = com_process.stderr.decode()
        logger.warning(f"System Execute Error: {errmsg}")
        raise Exception(errmsg)


def remove_enviroment(conda_env):
    system_execute(args=["conda", "env", "remove", "-n", f"{conda_env}"])
    logger.info(f"The learnware conda env [{conda_env}] is removed.")


def install_environment(learnware_dirpath, conda_env):
    """Install environment of a learnware

    Parameters
    ----------
    learnware_dirpath : str
        Path of the learnware folder
    conda_env : str
        a new conda environment will be created with the given name;

    Raises
    ------
    Exception
        Lack of the environment configuration file.
    """
    with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
        logger.info(f"learnware_dir namelist: {os.listdir(learnware_dirpath)}")
        if "environment.yaml" in os.listdir(learnware_dirpath):
            yaml_path: str = os.path.join(learnware_dirpath, "environment.yaml")
            yaml_path_filter: str = os.path.join(tempdir, "environment_filter.yaml")
            logger.info(f"checking the avaliabe conda packages for {conda_env}")
            filter_nonexist_conda_packages_file(yaml_file=yaml_path, output_yaml_file=yaml_path_filter)
            # create environment
            logger.info(f"create conda env [{conda_env}] according to .yaml file")
            system_execute(args=["conda", "env", "create", "--name", f"{conda_env}", "--file", f"{yaml_path_filter}"])

        elif "requirements.txt" in os.listdir(learnware_dirpath):
            requirements_path: str = os.path.join(learnware_dirpath, "requirements.txt")
            requirements_path_filter: str = os.path.join(tempdir, "requirements_filter.txt")
            logger.info(f"checking the avaliabe pip packages for {conda_env}")
            filter_nonexist_pip_packages_file(requirements_file=requirements_path, output_file=requirements_path_filter)
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
                ]
            )
        else:
            raise Exception("Environment.yaml or requirements.txt not found in the learnware folder.")

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
