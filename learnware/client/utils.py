import os
import zipfile
import tempfile

from . import package_utils
from .utils import system_execute


from ..logger import get_module_logger
from .package_utils import filter_nonexist_conda_packages_file, filter_nonexist_pip_packages_file

logger = get_module_logger(module_name="client_utils")


def system_execute(command):
    retcd: int = os.system(command=command)
    if retcd != 0:
        raise RuntimeError(f"Command {command} failed with return code {retcd}")


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
            logger.info(f"zip_file namelist: {z_file.namelist}")
            if "environment.yaml" in z_file.namelist():
                z_file.extract(member="environment.yaml", path=tempdir)
                yaml_path: str = os.path.join(tempdir, "environment.yaml")
                yaml_path_filter: str = os.path.join(tempdir, "environment_filter.yaml")
                filter_nonexist_conda_packages_file(yaml_file=yaml_path, output_yaml_file=yaml_path_filter)
                # create environment
                system_execute(command=f"conda env update --name {conda_env} --file {yaml_path_filter}")

            elif "requirements.txt" in z_file.namelist():
                z_file.extract(member="requirements.txt", path=tempdir)
                requirements_path: str = os.path.join(tempdir, "requirements.txt")
                requirements_path_filter: str = os.path.join(tempdir, "requirements_filter.txt")
                filter_nonexist_pip_packages_file(
                    requirements_file=requirements_path, output_file=requirements_path_filter
                )
                system_execute(command=f"conda create --name {conda_env}")
                system_execute(
                    command=f"conda run --no-capture-output python3 -m pip install -r {requirements_path_filter}"
                )
            else:
                raise Exception("Environment.yaml or requirements.txt not found in the learnware zip file.")


def remove_enviroment(conda_env):
    system_execute(command=f"conda env remove -n {conda_env}")
