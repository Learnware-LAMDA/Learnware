import numpy as np

class Specification:
    def __init__(self):
        self.property=None
        self.stat_spec = StatSpecification()
    
    def get_stat_spec(self):
        return self.stat_spec
    
    def get_property(self):
        return self.property

class StatSpecification:
    def __init__(self):
        self.Z = None
        self.beta = None
    
    def generate_stat_spec_from_raw(self, X: np.ndarray):
        pass

    def save(self, filepath: str = "./stat_spec.npy"):
        pass

    def load(self, filepath: str = "./stat_spec.npy"):
        pass