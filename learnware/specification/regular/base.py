from __future__ import annotations

from ..base import BaseStatSpecification

from torch.nn.functional import cosine_similarity


class RegularStatSpecification(BaseStatSpecification):
    def generate_stat_spec(self, **kwargs):
        self.generate_stat_spec_from_data(**kwargs)

    def generate_stat_spec_from_data(self, **kwargs):
        """Construct statistical specification from raw dataset
        - kwargs may include the feature, label and model
        - kwargs also can include hyperparameters of specific method for specifaction generation
        """
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")


class TaskVectorSpecification(RegularStatSpecification):
    
    @property
    def task_vector(self):
        raise NotImplemented

    def similarity(self, other: TaskVectorSpecification) -> float:
        """Compute cosine similarity between two task vectors.
        """
        v1, v2 = self.task_vector, other.task_vector
        
        return cosine_similarity(v1, v2, dim=0)

    def dist(self, other: BaseStatSpecification):
        v1, v2 = self.task_vector, other.task_vector
        
        similarity = cosine_similarity(v1, v2, dim=0)   # [-1, 1]
        return (-similarity + 1) / 2