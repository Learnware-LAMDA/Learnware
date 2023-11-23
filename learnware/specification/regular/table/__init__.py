from .utils import is_fast_pytorch_kmeans_available

from ....utils import is_torch_available
from ....logger import get_module_logger

logger = get_module_logger("regular_table_spec")

if not is_torch_available(verbose=False):
    RKMETableSpecification = None
    RKMEStatSpecification = None
    rkme_solve_qp = None
    logger.error(
        f"RKMETableSpecification, RKMEStatSpecification and rkme_solve_qp are skipped because 'torch' is not installed!"
    )
else:
    from .rkme import RKMETableSpecification, RKMEStatSpecification, rkme_solve_qp
