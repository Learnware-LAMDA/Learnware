from ....logger import get_module_logger
from ....utils import is_torch_available

logger = get_module_logger("regular_table_spec")

if not is_torch_available(verbose=False):
    RKMETableSpecification = None
    RKMEStatSpecification = None
    rkme_solve_qp = None
    logger.error(
        "RKMETableSpecification, RKMEStatSpecification and rkme_solve_qp are not available because 'torch' is not installed!"
    )
else:
    from .rkme import RKMEStatSpecification, RKMETableSpecification, rkme_solve_qp

__all__ = ["RKMEStatSpecification", "RKMETableSpecification", "rkme_solve_qp"]
