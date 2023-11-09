from ....logger import get_module_logger

logger = get_module_logger("regular_text_spec_utils")


def is_sentence_transformers_avaliable():
    try:
        import sentence_transformers
    except ModuleNotFoundError as err:
        logger.warning("ModuleNotFoundError: sentence_transformers is not installed, please install pytorch!")
        return False
    return True
