from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("market_hetero")

if not is_torch_available(verbose=False):
    HeteroMapTableOrganizer = None
    HeteroStatSearcher = None
    logger.error("HeteroMapTableOrganizer and HeteroStatSearcher are not available because 'torch' is not installed!")
else:
    from .organizer import HeteroMapTableOrganizer
    from .searcher import HeteroStatSearcher

__all__ = ["HeteroMapTableOrganizer", "HeteroStatSearcher"]
