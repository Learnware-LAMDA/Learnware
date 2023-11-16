from .organizer import AnchoredOrganizer
from .user_info import AnchoredUserInfo

from ...utils import is_torch_available
from ...logger import get_module_logger

logger = get_module_logger("market_anchor")

if not is_torch_available(verbose=False):
    AnchoredSearcher = None
    logger.warning("AnchoredSearcher is skipped because 'torch' is not installed!")
else:
    from .searcher import AnchoredSearcher
