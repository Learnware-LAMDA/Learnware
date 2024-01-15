from .organizer import EasyOrganizer
from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("market_easy")

if not is_torch_available(verbose=False):
    EasySearcher = None
    EasySemanticChecker = None
    EasyStatChecker = None
    EasyExactSemanticSearcher = None
    EasyFuzzSemanticSearcher = None
    EasyStatSearcher = None
    logger.error("EasySeacher and EasyChecker are not available because 'torch' is not installed!")
else:
    from .checker import EasySemanticChecker, EasyStatChecker
    from .searcher import EasyExactSemanticSearcher, EasyFuzzSemanticSearcher, EasySearcher, EasyStatSearcher

__all__ = [
    "EasyOrganizer",
    "EasySemanticChecker",
    "EasyStatChecker",
    "EasyExactSemanticSearcher",
    "EasyFuzzSemanticSearcher",
    "EasySearcher",
    "EasyStatSearcher",
]
