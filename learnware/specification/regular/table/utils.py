from ....logger import get_module_logger

logger = get_module_logger("regular_table_spec_utils")


def is_fast_pytorch_kmeans_available(verbose=False):
    try:
        import fast_pytorch_kmeans
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning(
                "ModuleNotFoundError: fast_pytorch_kmeans is not installed, please install fast_pytorch_kmeans!"
            )
        return False
    return True
