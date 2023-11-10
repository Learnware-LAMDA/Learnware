from ..logger import get_module_logger
from ..utils import is_torch_avaliable
from .utils import is_geatpy_avaliable, is_lightgbm_avaliable

logger = get_module_logger("reuse")

if not is_geatpy_avaliable(verbose=False):
    EnsemblePruningReuser = None
    logger.warning("EnsemblePruningReuser is skipped due to 'geatpy' is not installed!")
else:
    from .ensemble_pruning import EnsemblePruningReuser

if not is_torch_avaliable(verbose=False):
    AveragingReuser = None
    logger.warning("AveragingReuser is skipped due to 'torch' is not installed!")
else:
    from .averaging import AveragingReuser

if not is_lightgbm_avaliable(verbose=False) or not is_torch_avaliable(verbose=False):
    JobSelectorReuser = None
    uninstall_packages = [
        value
        for flag, value in zip(
            [
                is_lightgbm_avaliable(verbose=False),
                is_torch_avaliable(verbose=False),
            ],
            ["lightgbm", "torch"],
        )
        if flag is False
    ]
    logger.warning(f"JobSelectorReuser is skipped due to {uninstall_packages} is not installed!")
else:
    from .job_selector import JobSelectorReuser
