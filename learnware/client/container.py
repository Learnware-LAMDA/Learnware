import os
import docker
import pickle
import atexit
import tarfile
import tempfile
import shortuuid
from concurrent.futures import ThreadPoolExecutor

from typing import List, Union, Optional
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
            atexit.register(self.remove_env)
            self._init_env()
        self._setup_env_and_metadata()

    def remove_env(self):
        if self.cleanup_flag is True:
            try:
                self.cleanup_flag = False
                self._remove_env()
            except KeyboardInterrupt:
                self.cleanup_flag = True
                logger.warning("The KeyboardInterrupt is ignored when removing the container env!")
                self.remove_env()
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
    def __init__(self, model_config: dict, learnware_dirpath: str, conda_env: Optional[str] = None, build: bool = True):
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

            model_config = self.model_config.copy()
            model_config["module_path"] = Learnware.get_model_module_abspath(
                self.learnware_dirpath, model_config["module_path"]
            )
            with open(model_path, "wb") as model_fp:
                pickle.dump(model_config, model_fp)

            system_execute(
                [
                    "conda",
                    "run",
                    "-n",
                    f"{self.conda_env}",
                    "--no-capture-output",
                    "python",
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
        self.reset(input_shape=input_shape, output_shape=output_shape)

    def _run_model_with_script(self, method, **kargs):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            input_path = os.path.join(tempdir, "input.pkl")
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")

            model_config = self.model_config.copy()
            model_config["module_path"] = Learnware.get_model_module_abspath(
                self.learnware_dirpath, model_config["module_path"]
            )

            with open(model_path, "wb") as model_fp:
                pickle.dump(model_config, model_fp)

            with open(input_path, "wb") as input_fp:
                pickle.dump({"method": method, "kargs": kargs}, input_fp)

            system_execute(
                [
                    "conda",
                    "run",
                    "-n",
                    f"{self.conda_env}",
                    "--no-capture-output",
                    "python",
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
        self.env_script = os.path.join(C.package_path, "client", "scripts", "install_env.py")
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
            "pids_limit": -1,
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

        logger.info("Install learnware package in docker.")
        result = container.exec_run(
            " ".join(
                [
                    "conda",
                    "run",
                    "-n",
                    "base",
                    "--no-capture-output",
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "learnware",
                ]
            )
        )
        if result.exit_code != 0:
            logger.error(f"Install learnware package in docker failed!\n{result.output.decode('utf-8')}")

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

    @staticmethod
    def _change_path_to_container(path):
        file_name = os.path.basename(path)
        if "." in file_name:
            file_dir = os.path.basename(os.path.dirname(path))
            return f"/tmp/{file_dir}/{file_name}"
        else:
            return f"/tmp/{file_name}"

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

    def _install_environment(self, conda_env):
        """Install environment of a learnware in docker container

        Parameters
        ----------
        conda_env : str
            a new conda environment will be created with the given name

        Raises
        ------
        Exception
            Lack of the environment configuration file.
        """
        run_cmd_times = 10
        self.learnware_dirpath_container = self._change_path_to_container(self.learnware_dirpath)
        self._copy_file_to_container(self.learnware_dirpath, self.learnware_dirpath_container)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            env_script_container_path = self._change_path_to_container(os.path.join(tempdir, "install_env.py"))
            self._copy_file_to_container(self.env_script, env_script_container_path)

            success_flag = False
            logger.info("Install environment dependencies in docker.")
            for i in range(run_cmd_times):
                result = self.docker_container.exec_run(
                    " ".join(
                        [
                            "conda",
                            "run",
                            "-n",
                            "base",
                            "--no-capture-output",
                            "python",
                            f"{env_script_container_path}",
                            "--learnware-dirpath",
                            f"{self.learnware_dirpath_container}",
                            "--conda-env",
                            f"{conda_env}",
                        ]
                    )
                )
                if result.exit_code == 0:
                    success_flag = True
                    break
                else:
                    self.docker_container.exec_run("conda clean --all")
            if not success_flag:
                logger.error(f"Install environment dependencies in docker failed!\n{result.output.decode('utf-8')}")

    def _setup_env_and_metadata(self):
        """setup env and set the input and output shape by communicating with docker"""
        self._install_environment(self.conda_env)
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            output_path = os.path.join(tempdir, "output.pkl")
            model_path = os.path.join(tempdir, "model.pkl")
            self.docker_model_script_path = self._change_path_to_container(os.path.join(tempdir, "run_model.py"))

            output_path_container = self._change_path_to_container(output_path)
            model_path_container = self._change_path_to_container(model_path)

            docker_model_config = self.model_config.copy()
            docker_model_config["module_path"] = (
                self.learnware_dirpath_container + "/" + docker_model_config["module_path"]
            )

            with open(model_path, "wb") as model_fp:
                pickle.dump(docker_model_config, model_fp)
            self._copy_file_to_container(model_path, model_path_container)
            self._copy_file_to_container(self.model_script, self.docker_model_script_path)

            result = self.docker_container.exec_run(
                " ".join(
                    [
                        "conda",
                        "run",
                        "-n",
                        f"{self.conda_env}",
                        "--no-capture-output",
                        "python",
                        f"{self.docker_model_script_path}",
                        "--model-path",
                        f"{model_path_container}",
                        "--output-path",
                        f"{output_path_container}",
                    ]
                )
            )
            if result.exit_code != 0:
                logger.error(f"Instantiate learnware in docker failed!\n{result.output.decode('utf-8')}")
            self._copy_file_from_container(output_path_container, output_path)

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

            input_path_container = self._change_path_to_container(input_path)
            output_path_container = self._change_path_to_container(output_path)
            model_path_container = self._change_path_to_container(model_path)

            docker_model_config = self.model_config.copy()
            docker_model_config["module_path"] = (
                self.learnware_dirpath_container + "/" + docker_model_config["module_path"]
            )
            with open(model_path, "wb") as model_fp:
                pickle.dump(docker_model_config, model_fp)

            with open(input_path, "wb") as input_fp:
                pickle.dump({"method": method, "kargs": kargs}, input_fp)

            self._copy_file_to_container(model_path, model_path_container)
            self._copy_file_to_container(input_path, input_path_container)

            result = self.docker_container.exec_run(
                " ".join(
                    [
                        "conda",
                        "run",
                        "-n",
                        f"{self.conda_env}",
                        "--no-capture-output",
                        "python",
                        f"{self.docker_model_script_path}",
                        "--model-path",
                        f"{model_path_container}",
                        "--input-path",
                        f"{input_path_container}",
                        "--output-path",
                        f"{output_path_container}",
                    ]
                )
            )
            if result.exit_code != 0:
                logger.error(f"Run learnware in docker failed!\n{result.output.decode('utf-8')}")
            self._copy_file_from_container(output_path_container, output_path)

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
        cleanup=True,
        mode="conda",
        ignore_error=True,
    ):
        """The initializaiton method for base reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The learnware list to reuse and make predictions
        """
        if isinstance(learnwares, Learnware):
            learnwares = [learnwares]

        assert all(
            [isinstance(_learnware.get_model(), dict) for _learnware in learnwares]
        ), "the learnwre model should not be instantiated for reuser with containter"

        self.mode = mode
        assert self.mode in {"conda", "docker"}, f"mode must be in ['conda', 'docker'], should not be {self.mode}"
        self.learnware_list = learnwares
        self.cleanup = cleanup
        self.ignore_error = ignore_error

    @staticmethod
    def _destroy_docker_container(container):
        try:
            ModelDockerContainer._destroy_docker_container(container)
        except KeyboardInterrupt:
            logger.warning("The KeyboardInterrupt is ignored when removing the container env!")
            LearnwaresContainer._destroy_docker_container(container)
            
    def __enter__(self):
        if self.mode == "conda":
            self.learnware_containers = [
                Learnware(
                    _learnware.id,
                    ModelCondaContainer(_learnware.get_model(), _learnware.get_dirpath()),
                    _learnware.get_specification(),
                    _learnware.get_dirpath(),
                )
                for _learnware in self.learnware_list
            ]
        else:
            atexit.register(self._destroy_docker_container, self._docker_container)
            self._docker_container = ModelDockerContainer._generate_docker_container()
            self.learnware_containers = [
                Learnware(
                    _learnware.id,
                    ModelDockerContainer(
                        _learnware.get_model(), _learnware.get_dirpath(), self._docker_container, build=False
                    ),
                    _learnware.get_specification(),
                    _learnware.get_dirpath(),
                )
                for _learnware in self.learnware_list
            ]

        model_list = [_learnware.get_model() for _learnware in self.learnware_containers]
        if self.mode == "conda":
            with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
                results = executor.map(
                    self._initialize_model_container, model_list, [self.ignore_error] * len(model_list)
                )
            self.results = list(results)
        else:
            self.results = []
            for model_item in model_list:
                self.results.append(self._initialize_model_container(model_item, self.ignore_error))

        if sum(self.results) < len(self.learnware_list):
            logger.warning(
                f"{len(self.learnware_list) - sum(results)} of {len(self.learnware_list)} learnwares init failed! These learnwares will be ignored"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cleanup:
            self.learnware_containers = None
            self.results = None
            return

        model_list = [_learnware.get_model() for _learnware in self.learnware_containers]
        with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            executor.map(self._destroy_model_container, model_list, [self.ignore_error] * len(model_list))

        self.learnware_containers = None
        self.results = None

        if self.mode == "docker":
            ModelDockerContainer._destroy_docker_container(self._docker_container)

    @staticmethod
    def _initialize_model_container(model: ModelCondaContainer, ignore_error=True):
        try:
            model.init_and_setup_env()
        except Exception as err:
            if not ignore_error:
                raise err
            logger.warning(f"build env {model.conda_env} failed due to {err}")
            return False
        return True

    @staticmethod
    def _destroy_model_container(model: ModelCondaContainer, ignore_error=True):
        try:
            model.remove_env()
        except Exception as err:
            if not ignore_error:
                raise err
            logger.warning(f"remove env {model.conda_env} failed due to {err}")
            return False
        return True

    def get_learnware_flags(self):
        return self.results

    def get_learnwares_with_container(self):
        learnware_containers = [
            _learnware for _learnware, _result in zip(self.learnware_containers, self.results) if _result is True
        ]
        return learnware_containers
