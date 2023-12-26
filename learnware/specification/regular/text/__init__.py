from ....logger import get_module_logger
from ....utils import is_torch_available

logger = get_module_logger("regular_text_spec")

if not is_torch_available(verbose=False):
    RKMETextSpecification = None
    logger.error(f"RKMETextSpecification is not available because 'torch' is not installed!")
else:
    from .rkme import RKMETextSpecification
