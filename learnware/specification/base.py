import numpy as np


class BaseStatSpecification:
    def __init__(self):
        pass

    def generate_stat_spec_from_data(self, X: np.ndarray):
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")

    def save(self, filepath: str):
        raise NotImplementedError("save is not implemented")

    def load(self, filepath: str):
        raise NotImplementedError("load is not implemented")


class Specification:
    def __init__(self, semantic_spec=None):
        self.semantic_spec = semantic_spec
        self.stat_spec = BaseStatSpecification()  # stat_spec should be dict

    def get_stat_spec(self):
        return self.stat_spec

    def get_semantic_spec(self):
        return self.semantic_spec

    def update_stat_spec(self, name, new_stat_spec: BaseStatSpecification):
        self.stat_spec[name] = new_stat_spec

    def get_stat_spec_by_name(self, name: str):
        return self.stat_spec.get(name, None)
