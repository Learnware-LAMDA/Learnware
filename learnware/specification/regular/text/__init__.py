from .utils import is_sentence_transformers_avaliable

from ....utils import is_torch_avaliable
from ....logger import get_module_logger

logger = get_module_logger("regular_table_spec")

if not is_sentence_transformers_avaliable(verbose=True) or not is_torch_avaliable(verbose=False):
    RKMETextSpecification = None
    logger.warning("RKMETextSpecification is skipped because torch is not installed!")
else:
    from .rkme import RKMETextSpecification
