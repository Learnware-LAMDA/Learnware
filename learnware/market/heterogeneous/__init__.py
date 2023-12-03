from ...utils import is_torch_available
from ...logger import get_module_logger

logger = get_module_logger("market_hetero")

if not is_torch_available(verbose=False):
    HeteroMapTableOrganizer = None
    HeteroSearcher = None
    logger.error("HeteroMapTableOrganizer and HeteroSearcher are not available because 'torch' is not installed!")
else:
    from .organizer import HeteroMapTableOrganizer
    from .searcher import HeteroSearcher
