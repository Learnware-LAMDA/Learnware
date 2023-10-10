import os
import pickle
import tempfile
import shortuuid

from .utils import system_execute, install_environment, remove_enviroment
from ..config import C
from ..model.base import BaseModel

from ..logger import get_module_logger


logger = get_module_logger(module_name="client_container")

class ModelEnvContainer(BaseModel):
    
    def __init__(self, model_config: dict, learnware_zippath: str):
        """The initialization method for base model
        """
        
        self.model_script = os.path.join(C.package_path, 'learnware', 'client', 'run_model.py')
        self.model_config = model_config
        self.conda_env = f"learnware_{shortuuid.uuid()}"
        self.learnware_zippath = learnware_zippath
        install_environment(learnware_zippath, self.conda_env)
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            output_path = os.path.join(tempdir, 'output.pkl')
            model_path = os.path.join(tempdir, 'model.pkl')
            
            with open(model_path, 'wb') as model_fp:
                pickle.dump(model_config, model_fp)
            
            system_execute(f"conda run --no-capture-output python3 {self.model_script} --model-path {model_path} --output-path {output_path}")

            with open(output_path, 'rb') as output_fp:
                output_results = pickle.load(output_fp)
            
        if output_results['status'] != 'success':
            raise output_results['error_info']
        
        input_shape = output_results['metadata']['input_shape']
        output_shape = output_results['metadata']['output_shape']
            
        super(ModelEnvContainer, self).__init__(input_shape, output_shape)
    

    def run_model_with_script(self, method, **kargs):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            input_path = os.path.join(tempdir, 'input.pkl')
            output_path = os.path.join(tempdir, 'output.pkl')
            model_path = os.path.join(tempdir, 'model.pkl')
            
            with open(model_path, 'wb') as model_fp:
                pickle.dump(self.model_config, model_fp)
                
            with open(input_path, 'wb') as input_fp:
                pickle.dump({'method': method, 'kargs': kargs}, input_fp)

            system_execute(f"conda run --no-capture-output python3 {self.model_script} --model-path {model_path} --input-path {input_path} --output-path {output_path}")
            
            with open(output_path, 'rb') as output_fp:
                output_results = pickle.load(output_fp)
            
        if output_results['status'] != 'success':
            raise output_results['error_info']
        
        return output_results[output_results]
    
    def fit(self, X, y):
        self.run_model_with_script("fit", X=X, y=y)
    
    def predict(self, X):
        return self.run_model_with_script("predict", X=X)
    
    def finetune(self, X, y):
        self.run_model_with_script("finetune", X=X, y=y)
    
