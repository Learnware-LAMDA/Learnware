from .utils import is_torch_optimizer_avaliable
from ....utils import is_torch_avaliable
from ....logger import get_module_logger


logger = get_module_logger("regular_image_spec")

if not is_torch_optimizer_avaliable(verbose=True) or not is_torch_avaliable(verbose=False):
    RKMEImageSpecification = None
    logger.warning("RKMEImageSpecification is skipped because torch or torch-optimizer is not installed!")
else:
    from .rkme import RKMEImageSpecification
