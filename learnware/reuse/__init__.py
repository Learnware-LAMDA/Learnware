from ..logger import get_module_logger
from ..utils import is_torch_avaliable
from .utils import is_geatpy_avaliable, is_lightgbm_avaliable

logger = get_module_logger("reuse")

if not is_geatpy_avaliable(verbose=True):
    EnsemblePruningReuser = None
    logger.warning("EnsemblePruningReuser is skipped due to 'geatpy' is not installed!")
else:
    from .ensemble_pruning import EnsemblePruningReuser

if not is_torch_avaliable(verbose=False):
    AveragingReuser = None
    logger.warning("AveragingReuser is skipped due to 'torch' is not installed!")
else:
    from .averaging import AveragingReuser

if not is_lightgbm_avaliable(verbose=True) or not is_torch_avaliable(verbose=False):
    JobSelectorReuser = None
    logger.warning("JobSelectorReuser is skipped due to 'torch' is not installed!")
else:
    from .job_selector import JobSelectorReuser
