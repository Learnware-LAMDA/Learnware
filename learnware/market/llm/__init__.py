from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("market_llm")

if not is_torch_available(verbose=False):
    LLMEasyOrganizer = None
    LLMStatSearcher = None
    logger.error("LLMStatSearcher and LLMEasyOrganizer are not available because 'torch' is not installed!")
else:
    from .organizer import LLMEasyOrganizer
    from .searcher import LLMStatSearcher

__all__ = ["LLMEasyOrganizer", "LLMStatSearcher"]
