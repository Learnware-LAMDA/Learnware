from .base import BaseReuser
from .align import AlignLearnware

from ..logger import get_module_logger
from ..utils import is_torch_available

logger = get_module_logger("reuse")

if not is_torch_available(verbose=False):
    AveragingReuser = None
    FeatureAugmentReuser = None
    HeteroMapAlignLearnware = None
    FeatureAlignLearnware = None
    JobSelectorReuser = None
    EnsemblePruningReuser = None
    logger.error(
        "[AveragingReuser, FeatureAugmentReuser, HeteroMapAlignLearnware, FeatureAlignLearnware, JobSelectorReuser, EnsemblePruningReuser] are not available due to 'torch' is not installed!"
    )
else:
    from .averaging import AveragingReuser
    from .feature_augment import FeatureAugmentReuser
    from .hetero import HeteroMapAlignLearnware, FeatureAlignLearnware
    from .job_selector import JobSelectorReuser
    from .ensemble_pruning import EnsemblePruningReuser