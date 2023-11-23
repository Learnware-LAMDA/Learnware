from .utils import is_torch_optimizer_available, is_torchvision_available
from ....utils import is_torch_available
from ....logger import get_module_logger


logger = get_module_logger("regular_image_spec")

if not is_torch_available(verbose=False):
    RKMEImageSpecification = None
    logger.error(f"RKMEImageSpecification is skipped because 'torch' is not installed!")
else:
    from .rkme import RKMEImageSpecification
