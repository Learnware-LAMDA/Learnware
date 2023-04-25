import copy
import numpy as np


class BaseStatSpecification:
    """The Statistical Specification Interface, which provide save and load method"""

    def generate_stat_spec_from_data(self, **kwargs):
        """Construct statistical specification from raw dataset
        - kwargs may include the feature, label and model
        - kwargs also can include hyperparameters of specific method for specifaction generation
        """
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")

    def save(self, filepath: str):
        """Save the statistical specification into file in filepath

        Parameters
        ----------
        filepath : str
            The saved file path
        """
        raise NotImplementedError("save is not implemented")

    def load(self, filepath: str):
        """Load the statistical specification from file

        Parameters
        ----------
        filepath : str
            The file path to load

        """
        raise NotImplementedError("load is not implemented")


class Specification:
    """The specification interface, which manages the semantic specifications and statistical specifications"""

    def __init__(self, semantic_spec: dict = None, stat_spec: dict = None):
        """The initialization method

        Parameters
        ----------
        semantic_spec : dict, optional
            The initiailzed semantic specification, by default None
        stat_spec : dict, optional
            The initiailzaed statistical specification, by default None
        """
        self.semantic_spec = semantic_spec
        self.stat_spec = {} if stat_spec is None else stat_spec

    def __repr__(self) -> str:
        return "{}(Semantic Specification: {}, Statistical Specification: {})".format(
            type(self).__name__, type(self.semantic_spec).__name__, self.stat_spec
        )

    def get_stat_spec(self) -> dict:
        return self.stat_spec

    def get_semantic_spec(self) -> dict:
        return self.semantic_spec

    def upload_semantic_spec(self, semantic_spec: dict):
        """Upload or update semantic specification

        Parameters
        ----------
        semantic_spec : dict
            The new sementic specifications
        """
        self.semantic_spec = semantic_spec

    def update_stat_spec(self, **kwargs):
        """Update the statistical specification by the way of 'name'='value'"""
        for _k, _v in kwargs.items():
            self.stat_spec[_k] = _v

    def get_stat_spec_by_name(self, name: str) -> BaseStatSpecification:
        """Get statistical specification by its name

        Parameters
        ----------
        name : str
            The name of statistical specification

        Returns
        -------
        BaseStatSpecification
            The corresponding statistical specification w.r.t name
        """
        return self.stat_spec.get(name, None)
