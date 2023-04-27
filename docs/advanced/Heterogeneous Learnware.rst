==============================
Heterogeneous learnware
==============================

The learnware market naturally consists of models with different feature spaces, different label spaces, or different objectives. It is beneficial for the market to accommodate these heterogeneous learnwares and provide corresponding learnware recommendation and reuse services to the user so as to expand the applicable scope of learnware paradigm.

Models are submitted to the market with their original specifications. However, these specifications are hard to be used for responding to user requirements due to heterogeneity. Specifications of heterogeneous models reside in different specification spaces. The market needs to merge these specification spaces into a unified one. To achieve this adjustment, you need to implement the class ``EvolvedMarket``, especially the function ``EvolvedMarket.generate_new_stat_specification``, which generates new statistical specifcation in an identical space for each submitted model.

One important case is that models have different feature spaces. In order to enable the learnware market to handle heterogeneous feature spaces, you need to implement the class ``HeterogeneousFeatureMarket`` in the following way:

- First, design a method for the market to connect different feature spaces to a common subspace and implement the function ``HeterogeneousFeatureMarket.learn_mapping_functions``. This function uses specifications of all submitted models to learn mapping functions that can map the data in the original feature space to the common subspace and vice verse.
- Second, use learned mapping functions to implement the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original``.
- Third, use the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original`` to overwrite the mehtod ``EvolvedMarket.generate_new_stat_specification`` and  ``EvolvedMarket.EvolvedMarket.evolve_learnware_list`` of the base class ``EvolvedMarket``.