from ...logger import get_module_logger

logger = get_module_logger("llm_utils")


def is_llm(stat_specs: dict, semantic_spec: dict, verbose=True) -> bool:
    """Check if user_info satifies all the criteria required for enabling llm learnware search

    Parameters
    ----------
    user_info : BaseUserInfo
        user_info contains semantic_spec and stat_info

    Returns
    -------
    bool
        A flag indicating whether llm search is enabled for user_info
    """
    try:
        if "TaskVectorSpecification" not in stat_specs:
            if verbose:
                logger.warning("TaskVectorSpecification is not provided in stat_info.")
            return False
        
        semantic_data_type = semantic_spec["Data"]["Values"]
        if len(semantic_data_type) > 0 and semantic_data_type != ["Text"]:
            logger.warning("User doesn't provide correct data type, it must be Text.")
            return False

        semantic_task_type = semantic_spec["Task"]["Values"]
        if len(semantic_task_type) > 0 and semantic_task_type != ["Text Generation"]:
            logger.warning("User doesn't provide correct task type, it must be Text Generation.")
            return False

        return True
    except Exception as err:
        if verbose:
            logger.warning("Invalid llm search information provided.")
        return False
