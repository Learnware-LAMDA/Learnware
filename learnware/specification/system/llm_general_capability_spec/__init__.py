from ....logger import get_module_logger
from ....utils import is_torch_available

logger = get_module_logger("system_general_capability_spec")

if not is_torch_available(verbose=False):
    LLMGeneralCapabilitySpecification = None
    logger.error(
        "LLMGeneralCapabilitySpecification are not available because 'torch' is not installed!"
    )
else:
    from .spec import LLMGeneralCapabilitySpecification

__all__ = ["LLMGeneralCapabilitySpecification"]