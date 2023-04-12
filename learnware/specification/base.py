import copy
import numpy as np


class BaseStatSpecification:
    def __init__(self):
        pass

    def generate_stat_spec_from_data(self, **kwargs):
        """Construct statistical specification from raw dataset

        - kwargs may include the feature, label and model
        - kwargs also can include hyperparameters of specific method for specifaction generation
        """
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")

    def save(self, filepath: str):
        raise NotImplementedError("save is not implemented")

    def load(self, filepath: str):
        raise NotImplementedError("load is not implemented")


class Specification:
    def __init__(self, semantic_spec: dict = None, stat_spec: dict = None):
        self.semantic_spec = semantic_spec
        self.stat_spec = {} if stat_spec is None else stat_spec

    def __repr__(self) -> str:
        return "{}(Semantic Specification: {}, Statistical Specification: {})".format(
            type(self).__name__, type(self.semantic_spec).__name__, self.stat_spec
        )

    def get_stat_spec(self):
        return self.stat_spec

    def get_semantic_spec(self):
        return self.semantic_spec

    def upload_semantic_spec(self, semantic_spec: dict):
        self.semantic_spec = semantic_spec

    def update_stat_spec(self, **kwargs):
        for _k, _v in kwargs.items():
            self.stat_spec[_k] = _v

    def get_stat_spec_by_name(self, name: str):
        return self.stat_spec.get(name, None)
