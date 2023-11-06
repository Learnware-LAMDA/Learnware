from __future__ import annotations

from ..base import BaseStatSpecification
from loguru import logger


class SystemStatsSpecification(BaseStatSpecification):
    def generate_stat_spec(self, **kwargs):
        self.generate_stat_spec_from_system(**kwargs)

    def generate_stat_spec_from_system(self, **kwargs):
        """Construct statistical specification from raw dataset
        - kwargs may include the feature, label and model
        - kwargs also can include hyperparameters of specific method for specifaction generation
        """
        raise NotImplementedError("generate_stat_spec_from_data is not implemented")