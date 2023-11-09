from .utils import is_sentence_transformers_avaliable

from ....utils import is_torch_avaliable
from ....logger import get_module_logger

logger = get_module_logger("regular_text_spec")

if not is_sentence_transformers_avaliable(verbose=False) or not is_torch_avaliable(verbose=False):
    RKMETextSpecification = None
    uninstall_packages = [
        value
        for flag, value in zip(
            [
                is_sentence_transformers_avaliable(verbose=False),
                is_torch_avaliable(verbose=False),
            ],
            ["sentence_transformers", "torch"],
        )
        if flag is False
    ]
    logger.warning(f"RKMETextSpecification is skipped because {uninstall_packages} is not installed!")
else:
    from .rkme import RKMETextSpecification
