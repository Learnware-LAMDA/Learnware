from .utils import is_sentence_transformers_available
from ..table.utils import is_fast_pytorch_kmeans_available

from ....utils import is_torch_available
from ....logger import get_module_logger

logger = get_module_logger("regular_text_spec")

if not is_torch_available(verbose=False):
    RKMETextSpecification = None
    logger.warning(f"RKMETextSpecification is skipped because 'torch' is not installed!")
else:
    from .rkme import RKMETextSpecification
