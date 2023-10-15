import os
import pickle
import atexit
import tempfile
import shortuuid
from concurrent.futures import ProcessPoolExecutor

from typing import List, Union
from .utils import system_execute, install_environment, remove_enviroment
from ..config import C
from ..learnware import Learnware
from ..model.base import BaseModel

from ..logger import get_module_logger


logger = get_module_logger(module_name="client_container")


class ModelEnvContainer(BaseModel):
    def __init__(self, model_config: dict, learnware_zippath: str):
        self.model_script = os.path.join(C.package_path, "client", "scripts", "run_model.py")
        self.model_config = model_config
        self.conda_env = f"learnware_{shortuuid.uuid()}"
        self.learnware_zippath = learnware_zippath

    def init_env_and_metadata(self):
        install_environment(self.learnware_zippath, self.conda_env)
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
        super(ModelEnvContainer, self).__init__(input_shape, output_shape)
        atexit.register(self.remove_env)

    def remove_env(self):
        if self.conda_env is not None:
            self.conda_env = None
            remove_enviroment(self.conda_env)
        
    def run_model_with_script(self, method, **kargs):
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
        self.run_model_with_script("fit", X=X, y=y)

    def predict(self, X):
        return self.run_model_with_script("predict", X=X)

    def finetune(self, X, y) -> None:
        self.run_model_with_script("finetune", X=X, y=y)


class LearnwaresContainer:
    def __init__(self, learnwares: Union[List[Learnware], Learnware], learnware_zippaths: Union[List[str], str], cleanup=True):
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
        self.learnware_list = [
            Learnware(
                _learnware.id, ModelEnvContainer(_learnware.get_model(), _zippath), _learnware.get_specification()
            )
            for _learnware, _zippath in zip(learnwares, learnware_zippaths)
        ]
        self.cleanup = cleanup

    def __enter__(self):
        model_list = [_learnware.get_model() for _learnware in self.learnware_list]
        with ProcessPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            executor.map(self._initialize_model_container, model_list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cleanup:
            logger.warning(f"Notice, the learnware container env is not clean up!")
            return
        model_list = [_learnware.get_model() for _learnware in self.learnware_list]
        with ProcessPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            executor.map(self._destroy_model_container, model_list)

    @staticmethod
    def _initialize_model_container(model: ModelEnvContainer):
        try:
            model.init_env_and_metadata()
        except Exception as err:
            logger.error(f"build env {model.conda_env} failed due to {err}")

    @staticmethod
    def _destroy_model_container(model: ModelEnvContainer):
        try:
            model.remove_env()
        except Exception as err:
            logger.error(f"remove env {model.conda_env} failed due to {err}")

    def get_learnwares_with_container(self):
        return self.learnware_list
