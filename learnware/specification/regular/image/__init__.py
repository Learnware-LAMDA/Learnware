from .utils import is_torch_optimizer_avaliable, is_torch_vision_avaliable
from ....utils import is_torch_avaliable
from ....logger import get_module_logger


logger = get_module_logger("regular_image_spec")

if (
    not is_torch_vision_avaliable(verbose=False)
    or not is_torch_optimizer_avaliable(verbose=False)
    or not is_torch_avaliable(verbose=False)
):
    RKMEImageSpecification = None
    uninstall_packages = [
        value
        for flag, value in zip(
            [
                is_torch_vision_avaliable(verbose=False),
                is_torch_optimizer_avaliable(verbose=False),
                is_torch_avaliable(verbose=False),
            ],
            ["torchvision", "torch-optimizer", "torch"],
        )
        if flag is False
    ]

    logger.warning(f"RKMEImageSpecification is skipped because {uninstall_packages} is not installed!")
else:
    from .rkme import RKMEImageSpecification
