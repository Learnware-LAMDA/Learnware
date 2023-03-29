import numpy as np
from ..specification import Specification
import os

class Learnware:
    def __init__(self, id:str, name:str, model_path:str, specification:Specification, desc:str):
        self.id = id
        self.name = name
        self.model_path = model_path
        self.specification = specification
        self.desc = desc
        assert os.path.exists(self.model_path), "Model File {} NOT Found".format(self.model_path)

    def get_model(self)->str:
        pass

    def get_specification(self):
        pass

    def get_info(self):
        pass

    def update(self):
        # Empty Interface.
        raise Exception("'update' Method NOT Implemented.")