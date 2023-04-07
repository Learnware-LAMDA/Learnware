import os
import numpy as np
from typing import Union

from ..specification import Specification, BaseStatSpecification
from ..model import BaseModel
from ..utils import get_module_by_module_path


class Learnware:
    def __init__(self, id: str, name: str, model: BaseModel, specification: Specification):
        self.id = id
        self.name = name
        self.model = model
        self.specification = specification

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

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
