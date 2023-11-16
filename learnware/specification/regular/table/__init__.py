from ....utils import is_torch_available
from ....logger import get_module_logger

logger = get_module_logger("regular_table_spec")

if not is_torch_available(verbose=False):
    RKMETableSpecification = None
    RKMEStatSpecification = None
    logger.warning("RKMETableSpecification is skipped because torch is not installed!")
else:
    from .rkme import RKMETableSpecification, RKMEStatSpecification, rkme_solve_qp
