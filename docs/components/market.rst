.. _market:
================================
Market
================================

The learnware market receives high-performance machine learning models from developers, incorporates them into the system, and provides services to users by identifying and reusing learnware to help users solve current tasks. Developers voluntarily submit various learnwares to the learnware market, and the market conducts quality checks and further organization of these learnwares. When users submit task requirements, the learnware market automatically selects whether to recommend a single learnware or a combination of multiple learnwares. 

The learnware market will receive various kinds of learnwares, and learnwares from different feature/label spaces form numerous islands of specifications. All these islands together constitute the "specification world" in the learnware market. The market should discover and establish connections between different islands, and then merge them into a unified specification world. This further organization of learnwares can make the learnware search among all learnwares, not just learnwares which has the same feature space and label space with the user's task requirements.

Framework
======================================

The market class is initialized with a organizer class, a searcher class, and a list of checker classes. 

The organizer class should be able to organize the learnware in the market. It should be able to add, delete, and update learnware. It should also be able to search for learnware based on user requirement.

The searcher class should be able to search for learnware based on user requirement. It should be able to search for learnware based on user requirement.

The checker class is used for checking the learnware in some standards. It should check the utility of a learnware and is supposed to return the status and a message related to the learnware's check result.


Current Markets
======================================

Easy Market
-------------

Hetero Market
--------------

The learnware market naturally consists of models with different feature spaces, different label spaces, or different objectives. It is beneficial for the market to accommodate these heterogeneous learnwares and provide corresponding learnware recommendation and reuse services to the user so as to expand the applicable scope of learnware paradigm.

Models are submitted to the market with their original specifications. However, these specifications are hard to be used for responding to user requirements due to heterogeneity. Specifications of heterogeneous models reside in different specification spaces. The market needs to merge these specification spaces into a unified one. To achieve this adjustment, you need to implement the class ``EvolvedMarket``, especially the function ``EvolvedMarket.generate_new_stat_specification``, which generates new statistical specifcation in an identical space for each submitted model.

One important case is that models have different feature spaces. In order to enable the learnware market to handle heterogeneous feature spaces, you need to implement the class ``HeterogeneousFeatureMarket`` in the following way:

- First, design a method for the market to connect different feature spaces to a common subspace and implement the function ``HeterogeneousFeatureMarket.learn_mapping_functions``. This function uses specifications of all submitted models to learn mapping functions that can map the data in the original feature space to the common subspace and vice verse.
- Second, use learned mapping functions to implement the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original``.
- Third, use the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original`` to overwrite the mehtod ``EvolvedMarket.generate_new_stat_specification`` and  ``EvolvedMarket.EvolvedMarket.evolve_learnware_list`` of the base class ``EvolvedMarket``.

Current Checkers
======================================
The checkers check a learnware object in different aspects, including environment configuration (``CondaChecker``), semantic specifications (``EasySemanticChecker``), and statistical specifications (``EasyStatChecker``). The ``__call__`` method of each checker is designed to be invoked as a function to conduct the respective checks on the learnware and return the outcomes. It defines three types of learnwares: INVALID_LEARNWARE denotes the learnware does not pass the check, NONUSABLE_LEARNWARE denotes the learnware pass the check but cannot make prediction due to some env dependency, USABLE_LEARWARE denotes the leanrware pass the check and can make prediction. Currently, we have three checkers, which are described below.


CondaChecker Class
------------------
This checker checks a the environment of the learnware object. It creates a ``LearnwaresContainer`` instance to handle the Learnware and uses ``inner_checker`` to check the Learnware. If an exception occurs, it logs the error and returns ``BaseChecker.NONUSABLE_LEARNWARE`` status and error message.

EasySemanticChecker Class
-------------------------
This checker checks the semantic specification of a learnware object. It checks if the given semantic specification conforms to predefined standards. It verifies each key in predefined dictionary. If the check fails, it logs the error and returns ``NONUSABLE_LEARNWARE`` status and error message.

EasyStatChecker Class
---------------------
This checker checks the statistical specification and functionality of a learnware object. It performs multiple checks to validate the learnware. It checks for model instantiation, verifies input shape and statistical specifications, and test output shape using random generated data. In case of any exceptions, it logs the error and returns ``NONUSABLE_LEARNWARE`` status and error message.