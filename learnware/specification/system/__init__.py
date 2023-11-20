from .base import SystemStatSpecification
from ...utils import is_torch_available
from ...logger import get_module_logger

logger = get_module_logger("system_spec")

if not is_torch_available(verbose=False):
    HeteroMapTableSpecification = None
    logger.warning("HeteroMapTableSpecification is skipped because torch is not installed!")
else:
    from .hetero_table import HeteroMapTableSpecification
