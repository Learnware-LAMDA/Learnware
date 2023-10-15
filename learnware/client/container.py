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

class ModelContainer(BaseModel):
    def __init__(self, model_config: dict, learnware_zippath: str, build: bool=True):
        self.model_script = os.path.join(C.package_path, "client", "scripts", "run_model.py")
        self.model_config = model_config
        self.learnware_zippath = learnware_zippath
        self.build = build
        self.cleanup_flag = False
    
    def reset(self, **kwargs):
        for _k, _v in kwargs.items():
            if hasattr(self, _k):
                setattr(_k, _v)
                
    def init_env_and_set_metadata(self):
        """We must set `input_shape` and `output_shape` 
        """
        if self.build:
            self.cleanup_flag = True
            self._init_env()
            atexit.register(self.remove_env)
        
        self._set_metadata()
        
    def remove_env(self):
        if self.cleanup_flag is True:
            self.cleanup_flag = False
            try:
                self._remove_env()
            except Exception as err:
                self.cleanup_flag = True
                raise err
    
    def _set_metadata(self):
        raise NotImplementedError('_set_metadata method is not implemented!')
        
    def _init_env(self):
        raise NotImplementedError('_init_env method is not implemented!')
        
    def _remove_env(self):
        raise NotImplementedError('_remove_env method is not implemented!')
    
    def fit(self, X, y):
        raise NotImplementedError('fit method is not implemented!')

    def predict(self, X):
        raise NotImplementedError('predict method is not implemented!')

    def finetune(self, X, y) -> None:
        raise NotImplementedError('finetune method is not implemented!')
        
class ModelCondaContainer(ModelContainer):
    
    def __init__(self, model_config: dict, learnware_zippath: str, conda_env: str=None, build: bool=True):
        self.conda_env = f"learnware_{shortuuid.uuid()}" if conda_env is None else conda_env
        super(ModelCondaContainer, self).__init__(model_config, learnware_zippath, build)
        
    def _init_env(self):
        install_environment(self.learnware_zippath, self.conda_env)
    
    def _remove_env(self):
        remove_enviroment(self.conda_env)
        
    def _set_metadata(self):
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

class ModelDockerContainer(ModelCondaContainer):
    def __init__(self, model_config: dict, learnware_zippath: str, docker_img=None, conda_env: str=None, build: bool=True):
        """_summary_

        Parameters
        ----------
        build : bool, optional
            Whether to build the docker env, by default True
        """
        if docker_img is None:
            self.docker_img = None # create docker img
        self.conda_env = f"learnware_{shortuuid.uuid()}" if conda_env is None else conda_env
        # call init method of parent of parent class
        super(ModelCondaContainer, self).__init__(model_config, learnware_zippath, True if docker_img is None else build)
        
    def _set_metadata(self):
        raise NotImplementedError('_set_metadata method is not implemented!')
        
    def _init_env(self):
        raise NotImplementedError('_init_env method is not implemented!')
        
    def _remove_env(self):
        raise NotImplementedError('_remove_env method is not implemented!')
    
    def fit(self, X, y):
        raise NotImplementedError('fit method is not implemented!')

    def predict(self, X):
        raise NotImplementedError('predict method is not implemented!')

    def finetune(self, X, y) -> None:
        raise NotImplementedError('finetune method is not implemented!')
        
class LearnwaresContainer:
    def __init__(
        self, learnwares: Union[List[Learnware], Learnware], learnware_zippaths: Union[List[str], str], cleanup=True, mode='conda'
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
        
        if mode == 'conda':
            self.learnware_list = [
                Learnware(
                    _learnware.id, ModelCondaContainer(_learnware.get_model(), _zippath), _learnware.get_specification()
                )
                for _learnware, _zippath in zip(learnwares, learnware_zippaths)
            ]
        elif mode == 'docker':
            docker_img = self._generate_docker_img()
            self.learnware_list = [
                Learnware(
                    _learnware.id, ModelDockerContainer(_learnware.get_model(), _zippath, docker_img), _learnware.get_specification()
                )
                for _learnware, _zippath in zip(learnwares, learnware_zippaths)
            ]
        
        else:
            raise ValueError(f"mode must be in ['conda', 'docker'], should not be {mode}")
        self.results = [True] * len(learnwares)
        self.cleanup = cleanup
        print('234', self.learnware_list)

    def _generate_docker_img():
        return None
        
    def __enter__(self):
        model_list = [_learnware.get_model() for _learnware in self.learnware_list]
        with ProcessPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            results = executor.map(self._initialize_model_container, model_list)
        self.results = list(results)
        print('234', self.results, sum(self.results), len(self.learnware_list))
        if sum(self.results) < len(self.learnware_list):
            logger.warning(f'{len(self.learnware_list) - sum(results)} of {len(self.learnware_list)} learnwares init failed! This learnware will be ignored')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cleanup:
            logger.warning(f"Notice, the learnware container env is not cleaned up!")
            return
        model_list = [_learnware.get_model() for _learnware in self.learnware_list]
        with ProcessPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
            executor.map(self._destroy_model_container, model_list)

    @staticmethod
    def _initialize_model_container(model: ModelCondaContainer):
        try:
            model.init_env_and_set_metadata()
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
        learnwares = [_learnware for _learnware, _result in zip(self.learnware_list, self.results) if _result is True]
        print('233', learnwares, list(self.results))
        return learnwares
