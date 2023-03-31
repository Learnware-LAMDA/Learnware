import os
import numpy as np
from typing import Union

from ..specification import Specification, BaseStatSpecification
from ..model import BaseModel
from ..utils import get_module_by_module_path


class Learnware:
    def __init__(self, id: str, name: str, model: Union[BaseModel, dict], specification: Specification, desc: str):
        self.id = id
        self.name = name
        self.model = self._import_model(model)
        self.specification = specification
        self.desc = desc
        assert os.path.exists(self.model_path), "Model File {} NOT Found".format(self.model_path)

    def _import_model(model: Union[BaseModel, dict]) -> BaseModel:
        """_summary_

        Parameters
        ----------
        model : Union[BaseModel, str]
            - If isinstance(model, dict), autoimport the model w.r.t the following format:
                model = {
                    "module_path": str,
                    "class_name": str
                }
            - If isinstance(model, BaseModel), return model directly
        Returns
        -------
        BaseModel
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(model, BaseModel):
            return model

        elif isinstance(model, str):
            model_module = get_module_by_module_path(os.path.join(model, "run.py"))
            return getattr(model_module,)
        else:
            raise TypeError("model must be BaseModel or dict")

    def get_model(self) -> BaseModel:
        return self.model

    def get_specification(self) -> Specification:
        return self.specification

    def get_info(self):
        return self.desc
    
    def update_stat_spec(self, name, new_stat_spec: BaseStatSpecification):
        self.specification.update_stat_spec(name, new_stat_spec)
    
    def update(self):
        # Empty Interface.
        raise NotImplementedError("'update' Method is NOT Implemented.")
