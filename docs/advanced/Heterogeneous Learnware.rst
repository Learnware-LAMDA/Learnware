==============================
Heterogeneous learnware
==============================

The learnware market naturally consits of models with different feature spaces, different label spaces or different objectives. It is beneficial for the market to accomendate these heterogeneous learnwares and  provide corresponding learnware recommendation and reuse service to the user, so as to expand the applicable scope of learnware paradigm.

Models are submitted to the market with their original specifications, however, these specifcations are hard to be used for responding to user requirement due to heterogenousity. Specifcations of heterogenenous models reside in different specification spaces, the market needs to merge these specification spaces to a unfied one. To achive this adjustment, you need to implement the class ``EvolvedMarket``, especially the the function ``EvolvedMarket.generate_new_stat_specification``, which generates new statistical specifcation in an identical space for each submitted model.

One important case is that models has different feature space. In order to enable learnware market to handle heterogeneous feature spaces, you need to implement the class ``HeterogeneousFeatureMarket`` in the following way:

- First, design a method for the market to connect different feature space to a common subspace and implement the fucntion ``HeterogeneousFeatureMarket.learn_mapping_functions``, this function use specifcations of all submitted models to learn mapping functions which can map the data in the original feature space to the common subspace and vice verse.
- Second, use learned mapping functions to implement the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original``.
- Third, use the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original`` to overwrite the mehtod ``EvolvedMarket.generate_new_stat_specification`` of the base class.