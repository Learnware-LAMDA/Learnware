from .base import BaseReuser
from .align import AlignLearnware

from ..logger import get_module_logger
from ..utils import is_torch_available
from .utils import is_geatpy_available, is_lightgbm_available

logger = get_module_logger("reuse")

if not is_geatpy_available(verbose=False):
    EnsemblePruningReuser = None
    logger.warning("EnsemblePruningReuser is skipped due to 'geatpy' is not installed!")
else:
    from .ensemble_pruning import EnsemblePruningReuser

if not is_torch_available(verbose=False):
    AveragingReuser = None
    FeatureAugmentReuser = None
    HeteroMapAlignLearnware = None
    FeatureAlignLearnware = None
    logger.warning(
        "[AveragingReuser, FeatureAugmentReuser, HeteroMapAlignLearnware, FeatureAlignLearnware] is skipped due to 'torch' is not installed!"
    )
else:
    from .averaging import AveragingReuser
    from .feature_augment import FeatureAugmentReuser
    from .hetero import HeteroMapAlignLearnware, FeatureAlignLearnware

if not is_lightgbm_available(verbose=False) or not is_torch_available(verbose=False):
    JobSelectorReuser = None
    uninstall_packages = [
        value
        for flag, value in zip(
            [
                is_lightgbm_available(verbose=False),
                is_torch_available(verbose=False),
            ],
            ["lightgbm", "torch"],
        )
        if flag is False
    ]
    logger.warning(f"JobSelectorReuser is skipped due to {uninstall_packages} is not installed!")
else:
    from .job_selector import JobSelectorReuser
