from ...logger import get_module_logger

logger = get_module_logger("hetero_utils")


def is_hetero(stat_specs: dict, semantic_spec: dict) -> bool:
    """Check if user_info satifies all the criteria required for enabling heterogeneous learnware search

    Parameters
    ----------
    user_info : BaseUserInfo
        user_info contains semantic_spec and stat_info

    Returns
    -------
    bool
        A flag indicating whether heterogeneous search is enabled for user_info
    """
    try:
        table_stat_spec = stat_specs["RKMETableSpecification"]
        table_input_shape = table_stat_spec.get_z().shape[1]

        semantic_task_type = semantic_spec["Task"]["Values"]
        if len(semantic_task_type) > 0 and semantic_task_type not in [["Classification"], ["Regression"]]:
            logger.warning("User doesn't provide correct task type, it must be either Classification or Regression.")
            return False

        semantic_input_description = semantic_spec["Input"]
        semantic_description_dim = int(semantic_input_description["Dimension"])
        semantic_decription_feature_num = len(semantic_input_description["Description"])

        if semantic_decription_feature_num <= 0:
            logger.warning("At least one of Input.Description in semantic spec should be provides.")
            return False

        if table_input_shape != semantic_description_dim:
            logger.warning("User data feature dimensions mismatch with semantic specification.")
            return False

        return True

    except Exception as e:
        logger.warning(f"Invalid heterogeneous search information provided due to {e}. Use homogeneous search instead.")
        return False
