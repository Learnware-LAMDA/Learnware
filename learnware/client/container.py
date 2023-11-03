import os
import docker
import pickle
import atexit
import tarfile
import zipfile
import tempfile
import shortuuid
from concurrent.futures import ThreadPoolExecutor

from typing import List, Union
from .utils import system_execute, install_environment, remove_enviroment
from ..config import C
from ..learnware import Learnware
from ..model.base import BaseModel
from .package_utils import filter_nonexist_conda_packages_file, filter_nonexist_pip_packages_file

from ..logger import get_module_logger


logger = get_module_logger(module_name="client_container")


class ModelContainer(BaseModel):
    def __init__(self, model_config: dict, learnware_dirpath: str, build: bool = True):
        self.model_script = os.path.join(C.package_path, "client", "scripts", "run_model.py")
        self.model_config = model_config
        self.learnware_dirpath = learnware_dirpath
        self.build = build
        self.cleanup_flag = False

    def __enter__(self):
        self.init_and_setup_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_env()

    def reset(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def init_and_setup_env(self):
        """We must set `input_shape` and `output_shape`"""
        if self.build:
            self.cleanup_flag = True
            self._init_env()
            atexit.register(self.remove_env)

        self._setup_env_and_metadata()

    def remove_env(self):
        if self.cleanup_flag is True:
            self.cleanup_flag = False
            try:
                self._remove_env()
            except Exception as err:
                self.cleanup_flag = True
                raise err

    def _setup_env_and_metadata(self):
        raise NotImplementedError("_setup_env_and_metadata method is not implemented!")

    def _init_env(self):
        raise NotImplementedError("_init_env method is not implemented!")

    def _remove_env(self):
        raise NotImplementedError("_remove_env method is not implemented!")

    def fit(self, X, y):
        raise NotImplementedError("fit method is not implemented!")

    def predict(self, X):
        raise NotImplementedError("predict method is not implemented!")

    def finetune(self, X, y) -> None:
        raise NotImplementedError("finetune method is not implemented!")


class ModelCondaContainer(ModelContainer):
    def __init__(self, model_config: dict, learnware_dirpath: str, conda_env: str = None, build: bool = True):
        self.conda_env = f"learnware_{shortuuid.uuid()}" if conda_env is None else conda_env
        super(ModelCondaContainer, self).__init__(model_config, learnware_dirpath, build)

    def _init_env(self):
        install_environment(self.learnware_dirpath, self.conda_env)

    def _remove_env(self):
        remove_enviroment(self.conda_env)

    def _setup_env_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")

            with open(model_path, "wb") as model_fp:
                pickle.dump(self.model_config, model_fp)

            system_execute(
                [
                    "conda",
                    "run",
                    "-n",
                    f"{self.conda_env}",
                    "--no-capture-output",
                    "python3",
                    f"{self.model_script}",
                    "--model-path",
                    f"{model_path}",
                    "--output-path",
                    f"{output_path}",
                ]
            )

            with open(output_path, "rb") as output_fp:
                output_results = pickle.load(output_fp)

        if output_results["status"] != "success":
            raise output_results["error_info"]

        input_shape = output_results["metadata"]["input_shape"]
        output_shape = output_results["metadata"]["output_shape"]
        logger.info(f"input_shape: {input_shape}, output_shape: {output_shape}")
        self.reset(input_shape=input_shape, output_shape=output_shape)

    def _run_model_with_script(self, method, **kargs):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            input_path = os.path.join(tempdir, "input.pkl")
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")

            with open(model_path, "wb") as model_fp:
                pickle.dump(self.model_config, model_fp)

            with open(input_path, "wb") as input_fp:
                pickle.dump({"method": method, "kargs": kargs}, input_fp)

            system_execute(
                [
                    "conda",
                    "run",
                    "-n",
                    f"{self.conda_env}",
                    "--no-capture-output",
                    "python3",
                    f"{self.model_script}",
                    "--model-path",
                    f"{model_path}",
                    "--input-path",
                    f"{input_path}",
                    "--output-path",
                    f"{output_path}",
                ]
            )

            with open(output_path, "rb") as output_fp:
                output_results = pickle.load(output_fp)

        if output_results["status"] != "success":
            raise output_results["error_info"]

        return output_results[method]

    def fit(self, X, y):
        self._run_model_with_script("fit", X=X, y=y)

    def predict(self, X):
        return self._run_model_with_script("predict", X=X)

    def finetune(self, X, y) -> None:
        self._run_model_with_script("finetune", X=X, y=y)


class ModelDockerContainer(ModelContainer):
    def __init__(
        self,
        model_config: dict,
        learnware_dirpath: str,
        docker_container: object = None,
        build: bool = True,
    ):
        """_summary_

        Parameters
        ----------
        build : bool, optional
            Whether to build the docker env, by default True
        """

        self.docker_container = docker_container
        self.conda_env = f"learnware_{shortuuid.uuid()}"
        self.docker_model_config = None
        self.docker_model_script_path = None
        # call init method of parent of parent class
        super(ModelDockerContainer, self).__init__(model_config, learnware_dirpath, build)

    @staticmethod
    def _generate_docker_container():
        client = docker.from_env()
        http_proxy = os.environ.get("http_proxy")
        https_proxy = os.environ.get("https_proxy")

        container_config = {
            "image": "continuumio/miniconda3",
            "network_mode": "host",
            "detach": True,
            "tty": True,
            "command": "bash",
            "environment": {"http_proxy": http_proxy, "https_proxy": https_proxy},
        }
        container = client.containers.run(**container_config)
        environment_cmd = [
            "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple",
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/",
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/",
            "conda config --set show_channel_urls yes",
        ]
        for _cmd in environment_cmd:
            container.exec_run(_cmd)
        return container

    @staticmethod
    def _destroy_docker_container(docker_container):
        if isinstance(docker_container, docker.models.containers.Container):
            client = docker.from_env()
            container_ids = [container.id for container in client.containers.list()]

            if docker_container.id in container_ids:
                docker_container.stop()
                docker_container.remove()
                logger.info("Docker container is stopped and removed.")
            else:
                logger.info("Docker container has already been removed.")
        else:
            logger.error("Type of docker_container is not docker.models.containers.Container.")

    def _copy_file_to_container(self, local_path, container_path):
        directory_path = os.path.dirname(container_path)
        container_name = os.path.basename(container_path)
        self.docker_container.exec_run(f"mkdir -p {directory_path}")

        with tempfile.TemporaryDirectory(prefix="learnware_tar_") as tempdir:
            # Create a temporary tar file
            tar_file_path = os.path.join(tempdir, container_name + ".tar")
            with tarfile.open(tar_file_path, "w") as tar:
                tar.add(local_path, arcname=container_name)

            # Transfer the tar file to container
            with open(tar_file_path, "rb") as file_data:
                self.docker_container.put_archive(directory_path, file_data.read())

    def _copy_file_from_container(self, container_path, local_path):
        try:
            data, stat = self.docker_container.get_archive(container_path)
            tar_local_file = local_path + ".tar"
            with open(tar_local_file, "wb") as f:
                for chunk in data:
                    f.write(chunk)
            with tarfile.open(tar_local_file, "r") as tar:
                tar.extractall(os.path.dirname(tar_local_file))
            os.remove(tar_local_file)
        except docker.errors.NotFound as err:
            logger.error(f"Copy file from container failed due to {err}")

    def _install_environment(self, zip_path, conda_env):
        """Install environment of a learnware in docker container

        Parameters
        ----------
        zip_path : str
            Path of the learnware zip file
        conda_env : str
            a new conda environment will be created with the given name

        Raises
        ------
        Exception
            Lack of the environment configuration file.
        """
        run_cmd_times = 10
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            with zipfile.ZipFile(file=zip_path, mode="r") as z_file:
                success_flag = False
                logger.info(f"zip_file namelist: {z_file.namelist()}")

                if "environment.yaml" in z_file.namelist():
                    z_file.extract(member="environment.yaml", path=tempdir)
                    yaml_path: str = os.path.join(tempdir, "environment.yaml")
                    yaml_path_filter: str = os.path.join(tempdir, "environment_filter.yaml")
                    logger.info(f"checking the avaliabe conda packages for {conda_env}")
                    filter_nonexist_conda_packages_file(yaml_file=yaml_path, output_yaml_file=yaml_path_filter)
                    self._copy_file_to_container(yaml_path_filter, yaml_path_filter)

                    # create environment
                    logger.info(f"Create and update conda env [{conda_env}] according to .yaml file")
                    for i in range(run_cmd_times):
                        result = self.docker_container.exec_run(
                            " ".join(
                                ["conda", "env", "update", "--name", f"{conda_env}", "--file", f"{yaml_path_filter}"]
                            )
                        )
                        if result.exit_code == 0:
                            success_flag = True
                            break

                elif "requirements.txt" in z_file.namelist():
                    z_file.extract(member="requirements.txt", path=tempdir)
                    requirements_path: str = os.path.join(tempdir, "requirements.txt")
                    requirements_path_filter: str = os.path.join(tempdir, "requirements_filter.txt")
                    logger.info(f"checking the avaliabe pip packages for {conda_env}.")
                    filter_nonexist_pip_packages_file(
                        requirements_file=requirements_path, output_file=requirements_path_filter
                    )
                    logger.info(f"Create empty conda env [{conda_env}] in docker.")
                    for i in range(run_cmd_times):
                        result = self.docker_container.exec_run(
                            " ".join(["conda", "create", "-y", "--name", f"{conda_env}", "python=3.8"])
                        )
                        if result.exit_code == 0:
                            break
                    logger.info(f"install pip requirements for conda env [{conda_env}] in docker.")

                    self._copy_file_to_container(requirements_path_filter, requirements_path_filter)
                    for i in range(run_cmd_times):
                        result = self.docker_container.exec_run(
                            " ".join(
                                [
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
                        )
                        if result.exit_code == 0:
                            success_flag = True
                            break
                else:
                    raise Exception("Environment.yaml or requirements.txt not found in the learnware zip file.")

                if not success_flag:
                    logger.error(f"Install conda env [{conda_env}] in docker failed!")

        success_flag = False
        logger.info(f"Install learnware package for conda env [{conda_env}] in docker.")
        for i in range(run_cmd_times):
            result = self.docker_container.exec_run(
                " ".join(
                    [
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
            )
            if result.exit_code == 0:
                success_flag = True
                break

        if not success_flag:
            logger.error(f"Install learnware package for conda env [{conda_env}] in docker failed!")

    def _setup_env_and_metadata(self):
        """setup env and set the input and output shape by communicating with docker"""
        self._install_environment(self.learnware_dirpath, self.conda_env)
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")
            self.docker_model_script_path = os.path.join(tempdir, "run_model.py")

            docker_model_config = self.model_config.copy()
            with tempfile.TemporaryDirectory(prefix="learnware_model_config_") as config_tempdir:
                basename = os.path.basename(self.model_config["module_path"])
                docker_model_config["module_path"] = os.path.join(config_tempdir, basename)
                self._copy_file_to_container(
                    os.path.dirname(self.model_config["module_path"]),
                    os.path.dirname(docker_model_config["module_path"]),
                )

            self.docker_model_config = docker_model_config
            with open(model_path, "wb") as model_fp:
                pickle.dump(self.docker_model_config, model_fp)

            self._copy_file_to_container(model_path, model_path)
            self._copy_file_to_container(self.model_script, self.docker_model_script_path)
            self.docker_container.exec_run(
                " ".join(
                    [
                        "conda",
                        "run",
                        "-n",
                        f"{self.conda_env}",
                        "--no-capture-output",
                        "python3",
                        f"{self.docker_model_script_path}",
                        "--model-path",
                        f"{model_path}",
                        "--output-path",
                        f"{output_path}",
                    ]
                )
            )
            self._copy_file_from_container(output_path, output_path)

            with open(output_path, "rb") as output_fp:
                output_results = pickle.load(output_fp)

        input_shape = output_results["metadata"]["input_shape"]
        output_shape = output_results["metadata"]["output_shape"]
        logger.info(f"input_shape: {input_shape}, output_shape: {output_shape}")
        self.reset(input_shape=input_shape, output_shape=output_shape)

    def _init_env(self):
        """create docker container according to the str"""
        self.docker_container = ModelDockerContainer._generate_docker_container()

    def _remove_env(self):
        """remove the docker container"""
        ModelDockerContainer._destroy_docker_container(self.docker_container)

    def _run_model_with_script(self, method, **kargs):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            input_path = os.path.join(tempdir, "input.pkl")
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")

            with open(model_path, "wb") as model_fp:
                pickle.dump(self.docker_model_config, model_fp)

            with open(input_path, "wb") as input_fp:
                pickle.dump({"method": method, "kargs": kargs}, input_fp)

            self._copy_file_to_container(model_path, model_path)
            self._copy_file_to_container(input_path, input_path)

            self.docker_container.exec_run(
                " ".join(
                    [
                        "conda",
                        "run",
                        "-n",
                        f"{self.conda_env}",
                        "--no-capture-output",
                        "python3",
                        f"{self.docker_model_script_path}",
                        "--model-path",
                        f"{model_path}",
                        "--input-path",
                        f"{input_path}",
                        "--output-path",
                        f"{output_path}",
                    ]
                )
            )
            self._copy_file_from_container(output_path, output_path)

            with open(output_path, "rb") as output_fp:
                output_results = pickle.load(output_fp)

        if output_results["status"] != "success":
            raise output_results["error_info"]

        return output_results[method]

    def fit(self, X, y):
        """fit model by the communicating with docker"""
        self._run_model_with_script("fit", X=X, y=y)

    def predict(self, X):
        """predict model by the communicating with docker"""
        return self._run_model_with_script("predict", X=X)

    def finetune(self, X, y) -> None:
        """finetune model by the communicating with docker"""
        self._run_model_with_script("finetune", X=X, y=y)


class LearnwaresContainer:
    def __init__(
        self,
        learnwares: Union[List[Learnware], Learnware],
        learnware_zippaths: Union[List[str], str],
        cleanup=True,
        mode="conda",
    ):
        """The initializaiton method for base reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The learnware list to reuse and make predictions
        """
        if isinstance(learnwares, Learnware):
            learnwares = [learnwares]
        if isinstance(learnware_zippaths, str):
            learnware_zippaths = [learnware_zippaths]

        assert all(
            [isinstance(_learnware.get_model(), dict) for _learnware in learnwares]
        ), "the learnwre model should not be instantiated for reuser with containter"

        self.mode = mode
        assert self.mode in {"conda", "docker"}, f"mode must be in ['conda', 'docker'], should not be {self.mode}"
        self.learnware_list = learnwares
        self.learnware_zippaths = learnware_zippaths
        self.cleanup = cleanup

    def __enter__(self):
        if self.mode == "conda":
            self.learnware_containers = [
                Learnware(
                    _learnware.id, ModelCondaContainer(_learnware.get_model(), _zippath), _learnware.get_specification()
                )
                for _learnware, _zippath in zip(self.learnware_list, self.learnware_zippaths)
            ]
        else:
            self._docker_container = ModelDockerContainer._generate_docker_container()
            self.learnware_containers = [
                Learnware(
                    _learnware.id,
                    ModelDockerContainer(_learnware.get_model(), _zippath, self._docker_container, build=False),
                    _learnware.get_specification(),
                )
                for _learnware, _zippath in zip(self.learnware_list, self.learnware_zippaths)
            ]
            atexit.register(ModelDockerContainer._destroy_docker_container, self._docker_container)

        model_list = [_learnware.get_model() for _learnware in self.learnware_containers]
        with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            results = executor.map(self._initialize_model_container, model_list)
        self.results = list(results)

        if sum(self.results) < len(self.learnware_list):
            logger.warning(
                f"{len(self.learnware_list) - sum(results)} of {len(self.learnware_list)} learnwares init failed! This learnware will be ignored"
            )

        # if not self.cleanup and self.mode == "docker":
        #     _model_docker_container = self.learnware_containers[0].get_model()
        #     _model_docker_container.cleanup_flag = True
        #     atexit.register(_model_docker_container.remove_env)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cleanup:
            logger.warning(f"Notice, the learnware container env is not cleaned up!")
            self.learnware_containers = None
            self.results = None
            return

        model_list = [_learnware.get_model() for _learnware in self.learnware_containers]
        with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            executor.map(self._destroy_model_container, model_list)

        self.learnware_containers = None
        self.results = None

        if self.mode == "docker":
            ModelDockerContainer._destroy_docker_container(self._docker_container)

    @staticmethod
    def _initialize_model_container(model: ModelCondaContainer):
        try:
            model.init_and_setup_env()
        except Exception as err:
            logger.error(f"build env {model.conda_env} failed due to {err}")
            return False
        return True

    @staticmethod
    def _destroy_model_container(model: ModelCondaContainer):
        try:
            model.remove_env()
        except Exception as err:
            logger.error(f"remove env {model.conda_env} failed due to {err}")
            return False
        return True

    def get_learnwares_with_container(self):
        learnware_containers = [
            _learnware for _learnware, _result in zip(self.learnware_containers, self.results) if _result is True
        ]
        return learnware_containers
