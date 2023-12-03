import os
import tempfile
from shutil import copyfile
from typing import List, Tuple, Union, Optional

from ...utils import save_dict_to_yaml
from ...config import C

class LearnwareTemplate:
    def __init__(self):
        self.model_templates = {
            "pickle": {
                "class_name": 'PickleLoadedModel',
                "template_path": os.path.join(C.package_path, "tests", "templates", "pickle_model.py")
            }
        }
    
    def generate_requirements(self, filepath, requirements: Optional[List[Union[Tuple[str, str, str], str]]] = None):
        requirements = [] if requirements is None else requirements
        operators = {"==", "~=", ">=", "<=", ">", "<"}
        requirements_str = ""
        for requirement in requirements:
            if isinstance(requirement, str):
                line_str = requirement.strip() + "\n"
            elif isinstance(requirement, tuple):
                assert requirement[1] in operators, f"The operator of requirements is not supported."
                line_str = requirement[0].strip() + requirement[1].strip() + requirement[2].strip() + "\n"
            else:
                raise TypeError(f"requirement must be type str/tuple, rather than {type(requirement)}")
            
            requirements_str += line_str
            
        with open(filepath, "w") as fdout:
            fdout.write(requirements_str)
        
    def generate_learnware_yaml(self, filepath, model_config: Optional[dict] = None, stat_spec_config: Optional[List[dict]] = None):
        learnware_config = {}
        if model_config is not None:
            learnware_config["model"] = model_config
        if stat_spec_config is not None:
            learnware_config["stat_specifications"] = stat_spec_config

        save_dict_to_yaml(learnware_config, filepath)
        
    
    
    def generate_learnware_zipfile(
        self,
        learnware_zippath: str,
        model_template: str = "pickle",
        model_kwargs: Optional[dict] = None,
        stat_spec_config: Optional[List[dict]] = None,
        requirements: Optional[List[Union[Tuple[str, str, str], str]]] = None,
        **kwargs,
    ):
        with tempfile.TemporaryDirectory(suffix="learnware_template") as tempdir:
            requirement_filepath = os.path.join(tempdir, "requirements.txt")
            self.generate_requirements(requirement_filepath, requirements)
            
            model_filepath =  os.path.join(tempdir, "__init__.py")
            copyfile(self.model_templates[model_template]["template_path"], model_filepath)
            
            learnware_yaml_filepath = os.path.join(tempdir, "requirements.txt")
            model_config = {
                "class_name": self.model_templates[model_template]["class_name"],
                "kwargs": {} if model_kwargs is None else model_kwargs
            }
            self.generate_learnware_yaml(learnware_yaml_filepath, model_config, stat_spec_config)

            if model_template == "pickle":
                pickle_filepath = os.path.join(tempdir, model_config["kwargs"]["pickle_filepath"])
                copyfile(kwargs["pickle_filepath"], pickle_filepath)
                

    def generate_template_semantic_spec(self):
        pass