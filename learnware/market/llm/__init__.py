from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("llm")

if not is_torch_available(verbose=False):
    LLMSearcher = None
    logger.error("LLMSearcher are not available because 'torch' is not installed!")
else:
    # TODO
    pass

__all__ = ["LLMSearcher"]