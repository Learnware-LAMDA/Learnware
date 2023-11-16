from ....logger import get_module_logger

logger = get_module_logger("regular_text_spec_utils")


def is_sentence_transformers_available(verbose=False):
    try:
        import sentence_transformers
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning(
                "ModuleNotFoundError: sentence_transformers is not installed, please install sentence_transformers!"
            )
        return False
    return True
