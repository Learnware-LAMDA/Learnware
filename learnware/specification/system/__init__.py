from .base import SystemStatSpecification
from ...logger import get_module_logger
from ...utils import is_torch_available

logger = get_module_logger("system_spec")

if not is_torch_available(verbose=False):
    HeteroMapTableSpecification = None
    logger.error("HeteroMapTableSpecification is not available because 'torch' is not installed!")
else:
    from .hetero_table import HeteroMapTableSpecification

__all__ = ["SystemStatSpecification", "HeteroMapTableSpecification"]
