import numpy as np

class Specification:
    def __init__(self):
        self.property = None
        self.stat_spec = {} # stat_spec should be dict

    def get_stat_spec(self):
        return self.stat_spec

    def get_property(self):
        return self.property

    def update_stat_spec(self): # update specification method
        pass

class StatSpecification:
    def generate_stat_spec_from_data(self, X: np.ndarray):
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")

    def save(self, filepath: str = "./stat_spec.npy"):
        raise NotImplementedError("save is not implemented")


    def load(self, filepath: str = "./stat_spec.npy"):
        raise NotImplementedError("load is not implemented")
