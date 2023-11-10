from ....utils import is_torch_avaliable
from ....logger import get_module_logger

logger = get_module_logger("regular_table_spec")

if not is_torch_avaliable(verbose=False):
    RKMETableSpecification = None
    RKMEStatSpecification = None
    logger.warning("RKMETableSpecification is skipped because torch is not installed!")
else:
    from .rkme import RKMETableSpecification, RKMEStatSpecification
