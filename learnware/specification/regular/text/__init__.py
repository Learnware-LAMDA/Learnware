from ....logger import get_module_logger
from ....utils import is_torch_available

logger = get_module_logger("regular_text_spec")

if not is_torch_available(verbose=False):
    RKMETextSpecification = None
    GenerativeModelSpecification = None
    logger.error(
        "RKMETextSpecification and GenerativeModelSpecification are not available because 'torch' is not installed!"
    )
else:
    from .generative import GenerativeModelSpecification
    from .rkme import RKMETextSpecification

__all__ = ["RKMETextSpecification", "GenerativeModelSpecification"]
