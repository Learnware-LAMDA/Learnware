from .utils import is_torch_optimizer_available, is_torchvision_available
from ....utils import is_torch_available
from ....logger import get_module_logger


logger = get_module_logger("regular_image_spec")

if (
    not is_torchvision_available(verbose=False)
    or not is_torch_optimizer_available(verbose=False)
    or not is_torch_available(verbose=False)
):
    RKMEImageSpecification = None
    uninstall_packages = [
        value
        for flag, value in zip(
            [
                is_torchvision_available(verbose=False),
                is_torch_optimizer_available(verbose=False),
                is_torch_available(verbose=False),
            ],
            ["torchvision", "torch-optimizer", "torch"],
        )
        if flag is False
    ]

    logger.warning(f"RKMEImageSpecification is skipped because {uninstall_packages} is not installed!")
else:
    from .rkme import RKMEImageSpecification
