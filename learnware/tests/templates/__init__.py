import os
import tempfile
from dataclasses import dataclass, field
from shutil import copyfile
from typing import List, Tuple, Union, Optional

from ...utils import save_dict_to_yaml, convert_folder_to_zipfile
from ...config import C


@dataclass
class ModelTemplate:
    class_name: str = field(init=False)
    template_path: str = field(init=False)
    model_kwargs: dict = field(init=False)
@dataclass
class PickleModelTemplate(ModelTemplate):
    model_kwargs: dict
    pickle_filepath: str
    def __post_init__(self):
        self.class_name = "PickleLoadedModel"
        self.template_path = os.path.join(C.package_path, "tests", "templates", "pickle_model.py")
        default_model_kwargs = {
            "predict_method": "predict",
            "fit_method": "fit",
            "finetune_method": "finetune",
            "pickle_filename": "model.pkl",
        }
        default_model_kwargs.update(self.model_kwargs)
        self.model_kwargs = default_model_kwargs

@dataclass
class StatSpecTemplate:
    filepath: str
    type: str = field(default="RKMETableSpecification")
    
class LearnwareTemplate:

    @staticmethod
    def generate_requirements(filepath, requirements: Optional[List[Union[Tuple[str, str, str], str]]] = None):
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
    
    @staticmethod
    def generate_learnware_yaml(filepath, model_config: Optional[dict] = None, stat_spec_config: Optional[List[dict]] = None):
        learnware_config = {}
        if model_config is not None:
            learnware_config["model"] = model_config
        if stat_spec_config is not None:
            learnware_config["stat_specifications"] = stat_spec_config

        save_dict_to_yaml(learnware_config, filepath)
    
    @staticmethod
    def generate_learnware_zipfile(
        learnware_zippath: str,
        model_template: ModelTemplate,
        stat_spec_template: StatSpecTemplate,
        requirements: Optional[List[Union[Tuple[str, str, str], str]]] = None,
    ):
        with tempfile.TemporaryDirectory(suffix="learnware_template") as tempdir:
            requirement_filepath = os.path.join(tempdir, "requirements.txt")
            LearnwareTemplate.generate_requirements(requirement_filepath, requirements)
            
            model_filepath =  os.path.join(tempdir, "__init__.py")
            copyfile(model_template.template_path, model_filepath)
            
            learnware_yaml_filepath = os.path.join(tempdir, "learnware.yaml")
            model_config = {
                "class_name": model_template.class_name,
                "kwargs": model_template.model_kwargs,
            }
            
            stat_spec_config = {
                "module_path": "learnware.specification",
                "class_name": stat_spec_template.type,
                "file_name": "stat_spec.json",
                "kwargs": {}
            }
            copyfile(stat_spec_template.filepath, os.path.join(tempdir, stat_spec_config["file_name"]))
            LearnwareTemplate.generate_learnware_yaml(learnware_yaml_filepath, model_config, stat_spec_config=[stat_spec_config])
            
            if isinstance(model_template, PickleModelTemplate):
                pickle_filepath = os.path.join(tempdir, model_template.model_kwargs["pickle_filename"])
                copyfile(model_template.pickle_filepath, pickle_filepath)
                
            convert_folder_to_zipfile(tempdir, learnware_zippath)
