from ....logger import get_module_logger
from ....utils import is_torch_available

logger = get_module_logger("regular_image_spec")

if not is_torch_available(verbose=False):
    RKMEImageSpecification = None
    logger.error("RKMEImageSpecification is not available because 'torch' is not installed!")
else:
    from .rkme import RKMEImageSpecification
