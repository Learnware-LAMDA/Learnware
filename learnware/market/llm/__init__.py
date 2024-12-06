from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("market_llm")

if not is_torch_available(verbose=False):
    LLMStatSearcher = None
    logger.error("LLMStatSearcher is not available because 'torch' is not installed!")
else:
    from .searcher import LLMStatSearcher

__all__ = ["LLMStatSearcher"]
