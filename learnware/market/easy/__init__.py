from .organizer import EasyOrganizer

from ...utils import is_torch_avaliable
from ...logger import get_module_logger

logger = get_module_logger("market_easy")

if not is_torch_avaliable(verbose=False):
    EasySearcher = None
    EasySemanticChecker = None
    EasyStatChecker = None
    logger.warning("EasySeacher and EasyChecker are skipped because 'torch' is not installed!")
else:
    from .searcher import EasySearcher
    from .checker import EasySemanticChecker, EasyStatChecker
