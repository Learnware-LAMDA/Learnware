import os
import docker
import pickle
import atexit
import tempfile
import shortuuid
from concurrent.futures import ThreadPoolExecutor

from typing import List, Union
from .utils import system_execute, install_environment, remove_enviroment
from ..config import C
from ..learnware import Learnware
from ..model.base import BaseModel

from ..logger import get_module_logger


logger = get_module_logger(module_name="client_container")


class ModelContainer(BaseModel):
    def __init__(self, model_config: dict, learnware_zippath: str, build: bool = True):
        self.model_script = os.path.join(C.package_path, "client", "scripts", "run_model.py")
        self.model_config = model_config
        self.learnware_zippath = learnware_zippath
        self.build = build
        self.cleanup_flag = False

    def reset(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def init_and_setup_env(self):
        """We must set `input_shape` and `output_shape`"""
        if self.build:
            self.cleanup_flag = True
            self._init_env()
            atexit.register(self.reset_and_remove_env)

        self._setup_env_and_metadata()

    def reset_and_remove_env(self):
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
    def __init__(self, model_config: dict, learnware_zippath: str, conda_env: str = None, build: bool = True):
        self.conda_env = f"learnware_{shortuuid.uuid()}" if conda_env is None else conda_env
        super(ModelCondaContainer, self).__init__(model_config, learnware_zippath, build)

    def _init_env(self):
        install_environment(self.learnware_zippath, self.conda_env)

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
        print("input_shape", input_shape, "output_shape", output_shape)
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
        learnware_zippath: str,
        docker_img: object = None,
        build: bool = True,
    ):
        """_summary_

        Parameters
        ----------
        build : bool, optional
            Whether to build the docker env, by default True
        """

        self.docker_img = f"docker_img_{shortuuid.uuid()}" if docker_img is None else docker_img
        self.conda_env = f"learnware_{shortuuid.uuid()}"
        # call init method of parent of parent class
        super(ModelDockerContainer, self).__init__(model_config, learnware_zippath, build)

    @staticmethod
    def _generate_docker_container():
        client = docker.from_env()
        image = client.images.pull('continuumio/miniconda3')
        return client.containers.create(image)
    
    @staticmethod
    def _destroy_docker_container():
        # destroy
        pass

    def _setup_env_and_metadata(self):
        """setup env and set the input and output shape by communicating with docker"""
        raise NotImplementedError("_setup_env_and_metadata method is not implemented!")

    def _init_env(self):
        """create docker img according to the str self.docker_img, and creat the correponding conda python env"""
        client = docker.from_env()
        client.containers.run
        image, build_log = client.images.pull(path=image_path, tag=tag)
        raise NotImplementedError("_init_env method is not implemented!")

    def _remove_env(self):
        """remove the docker img"""
        raise NotImplementedError("_remove_env method is not implemented!")

    def fit(self, X, y):
        """fit model by the communicating with docker"""
        raise NotImplementedError("fit method is not implemented!")

    def predict(self, X):
        """predict model by the communicating with docker"""
        raise NotImplementedError("predict method is not implemented!")

    def finetune(self, X, y) -> None:
        """finetune model by the communicating with docker"""
        raise NotImplementedError("finetune method is not implemented!")


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
        print("234", self.learnware_list)

    def __enter__(self):
        if self.mode == "conda":
            self.learnware_containers = [
                Learnware(
                    _learnware.id, ModelCondaContainer(_learnware.get_model(), _zippath), _learnware.get_specification()
                )
                for _learnware, _zippath in zip(self.learnware_list, self.learnware_zippaths)
            ]
        else:
            self.docker_img = ModelDockerContainer._generate_docker_container()
            self.learnware_containers = [
                Learnware(
                    _learnware.id,
                    ModelDockerContainer(_learnware.get_model(), _zippath, self.docker_img, build=False),
                    _learnware.get_specification(),
                )
                for _learnware, _zippath in zip(self.learnware_list, self.learnware_zippaths)
            ]

        model_list = [_learnware.get_model() for _learnware in self.learnware_containers]
        with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            results = executor.map(self._initialize_model_container, model_list)
        self.results = list(results)

        if sum(self.results) < len(self.learnware_list):
            logger.warning(
                f"{len(self.learnware_list) - sum(results)} of {len(self.learnware_list)} learnwares init failed! This learnware will be ignored"
            )
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
            ModelDockerContainer._destroy_docker_container()

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
            model.reset_and_remove_env()
        except Exception as err:
            logger.error(f"remove env {model.conda_env} failed due to {err}")
            return False
        return True

    def get_learnwares_with_container(self):
        learnware_containers = [
            _learnware for _learnware, _result in zip(self.learnware_containers, self.results) if _result is True
        ]
        print("233", learnware_containers, list(self.results))
        return learnware_containers
