.. _market:

================================
Learnware Market
================================

The ``learnware market`` receives high-performance machine learning models from developers, incorporates them into the system, and provides services to users by identifying and reusing learnware to help users solve current tasks. Developers voluntarily submit various learnwares to the learnware market, and the market conducts quality checks and further organization of these learnwares. When users submit task requirements, the learnware market automatically selects whether to recommend a single learnware or a combination of multiple learnwares. 

The ``learnware market`` will receive various kinds of learnwares, and learnwares from different feature/label spaces form numerous islands of specifications. All these islands together constitute the ``specification world`` in the learnware market. The market should discover and establish connections between different islands, and then merge them into a unified specification world. This further organization of learnwares support search learnwares among all learnwares, not just among learnwares which has the same feature space and label space with the user's task requirements.

Framework
======================================

The ``learnware market`` is combined with a ``organizer``, a ``searcher``, and a list of ``checker``s. 

The ``organizer`` can store and organize learnwares in the market. It supports ``add``, ``delete``, and ``update`` operations for learnwares. It also provides the interface for ``searcher`` to search learnwares based on user requirement.

The ``searcher`` can search learnwares based on user requirement. The implementation of ``searcher`` is dependent on the concrete implementation and interface for ``organizer``, where usually an ``organizer`` can be compatible with multiple different ``searcher``s.

The ``checker`` is used for checking the learnware in some standards. It should check the utility of a learnware and is supposed to return the status and a message related to the learnware's check result. Only the learnwares who passed the ``checker`` could be able to be stored and added into the ``learnware market``. 



Current Checkers
======================================

The ``learnware`` package provide two different implementation of ``market`` where both of them share the same ``checker`` list. So we first introduce the details of ``checker``s.

The ``checker``s check a learnware object in different aspects, including environment configuration (``CondaChecker``), semantic specifications (``EasySemanticChecker``), and statistical specifications (``EasyStatChecker``). The ``__call__`` method of each checker is designed to be invoked as a function to conduct the respective checks on the learnware and return the outcomes. It defines three types of learnwares: ``INVALID_LEARNWARE`` denotes the learnware does not pass the check, ``NONUSABLE_LEARNWARE`` denotes the learnware pass the check but cannot make prediction, ``USABLE_LEARWARE`` denotes the leanrware pass the check and can make prediction. Currently, we have three ``checker``s, which are described below.


``CondaChecker``
------------------
This ``checker`` checks a the environment of the learnware object. It creates a ``LearnwaresContainer`` instance to handle the Learnware and uses ``inner_checker`` to check the Learnware. If an exception occurs, it logs the error and returns ``NONUSABLE_LEARNWARE`` status and error message.


``EasySemanticChecker``
-------------------------
This ``checker`` checks the semantic specification of a learnware object. It checks if the given semantic specification conforms to predefined standards. It verifies each key in predefined dictionary. If the check fails, it logs the error and returns ``NONUSABLE_LEARNWARE`` status and error message.


``EasyStatChecker``
---------------------

This ``checker`` checks the statistical specification and functionality of a learnware object. It performs multiple checks to validate the learnware. It checks for model instantiation, verifies input shape and statistical specifications, and test output shape using random generated data. In case of any exceptions, it logs the error and returns ``NONUSABLE_LEARNWARE`` status and error message.


Current Markets
======================================

The ``learnware`` package provide two different implementation of ``market``, i.e. ``Easy Market`` and ``Hetero Market``. They have different implementation of ``organizer`` and ``searcher``.


``Easy Market``
-----------------
Easy market is a basic realization of the learnware market. It consists of ``EasyOrganizer``, ``EasySearcher``, and the checker list ``[EasySemanticChecker, EasyStatChecker]``.


``Easy Organizer``
++++++++++++++++++++

``EasyOrganizer`` mainly has the following methods to store learnwares, which is an easy way to organize learnwares.

- **reload_market**: Reload the learnware market when server restarted, and return a flag indicating whether the market is reloaded successfully.
- **add_learnware**: Add a learnware with ``learnware_id``, ``semantic_spec`` and model files in ``zip_path`` into the market. Return the ``learnware_id`` and ``learnwere_status``. The ``learnwere_status`` is set ``check_status`` if it is provided, else ``checker`` will be called to generate the ``learnwere_status``.
- **delete_learnware**: Delete the learnware with ``id`` from the market, return a flag of whether the deletion is successfully.
- **update_learnware**: Update the learnware's ``zip_path``, ``semantic_spec``, ``check_status``. If None, the corresponding item is not updated. Return a flag indicating whether it passed the ``checker``.
- **get_learnwares**: Similar to **get_learnware_ids**, but return list of learnwares instead of ids.
- **reload_learnware**: Reload all the attributes of the learnware with ``learnware_id``.

``Easy Searcher``
++++++++++++++++++++

``EasySearcher`` consists of ``EasyFuzzsematicSearcher`` and ``EasyStatSearcher``. ``EasyFuzzsematicSearcher`` is a kind of ``Semantic Specification Searcher``, while ``EasyStatSearcher`` is a kind of ``Statistical Specification Searcher``. All these searchers return helpful learnwares based on ``BaseUserInfo`` provided by users.

``BaseUserInfo`` is a ``Python API`` for users to provide enough information to identify helpful learnwares.
When initializing ``BaseUserInfo``, three optional information can be provided: ``id``, ``semantic_spec`` and ``stat_info``. The introductions of these specifications is shown in `COMPONENTS: Specification <./spec.html>`_.


The semantic specification search and statistical specification search have been integrated into the same interface ``EasySearcher``. 

- **EasySearcher.__call__(self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**

  - It conducts the semantic seacher ``EasyFuzzsematicSearcher``  on all the learnwares from the ``organizer`` with the same ``check_status`` (All learnwares if ``check_status`` is None). If the result is not empty and the ``stat_info`` is provided in ``user_info``, then it conducts ``EasyStatSearcher``, and return the ``SearchResults``.


``Semantic Specification Searcher``
''''''''''''''''''''''''''''''''''''

``Semantic Specification Searcher`` is the first-stage search based on ``user_semantic``, identifying potentially helpful learnwares whose models solve tasks similar to your requirements. There are two types of Semantic Specification Search: ``EasyExactSemanticSearcher`` and ``EasyFuzzSemanticSearcher``. 

In these two searchers, each learnware in the ``learnware_list`` is compared with ``user_info`` according to their ``semantic_spec``, and added to the search result if mathched. Two semantic_spec are matched when all the key words are matched or empty in ``user_info``. Different keys have different matching rules. Their ``__call__`` functions are the same:

- **EasyExactSemanticSearcher/EasyFuzzSemanticSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo)-> SearchResults**

  - For keys ``Data``, ``Task``, ``Library`` and ``license``, two``semantic_spec`` keys are matched only if these values(only one value foreach key) of learnware ``semantic_spec`` exists in values(may be muliplevalues for one key) of user ``semantic_spec``.
  - For the key ``Scenario``, two ``semantic_spec`` keys are matched iftheir values have nonempty intersections.
  - For keys ``Name`` and ``Description``, the values are strings and caseis ignored. In ``EasyExactSemanticSearcher``, two ``semantic_spec`` keysare matched if these values of learnware ``semantic_spec`` is a substringof user ``semantic_spec``; In ``EasyFuzzSemanticSearcher``, first theexact semantic searcher is conducted like ``EasyExactSemanticSearcher``.If the result is empty, the fuzz semantic searcher is activated: the``learnware_list`` is sorted according to the fuzz score function ``fuzzpartial_ratio`` in ``rapidfuzz``.

The results are returned storing in ``single_results`` of ``SearchResults``.


``Statistical Specification Searcher``
''''''''''''''''''''''''''''''''''''''''''

If user's statistical specification ``stat_info`` is provided,  the learnware market can perform a more accurate leanware selection using ``EasyStatSearcher``. 

- **EasyStatSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**
 
  - It searches for helpful learnwares from ``learnware_list`` based on the ``stat_info`` in ``user_info``.
  - The result ``SingleSearchItem`` and ``MultipleSearchItem`` are both stored in ``SearchResults``. In ``SingleSearchItem``, it searches for single learnwares that could solve the user task; scores are also provided to represent the fitness of each single learnware and user task. In ``MultipleSearchItem``, it searches for a mixture of learnwares that could solve the user task better; the mixture learnware list and a score for the mixture is returned.
  - The parameter ``search_method`` provides two choice of search strategies for mixture learnwares: ``greedy`` and ``auto``. For the search method ``greedy``, each time it chooses a learnware to make their mixture closer to the user's ``stat_info``; for the search method ``auto``, it directly calculates a best mixture weight for the ``learnware_list``.
  - For single learnware search, we only return the learnwares with score larger than 0.6; For multiple learnware search, the parameter ``max_search_num`` specifies the maximum length of the returned mixture learnware list. 


``Easy Checker``
++++++++++++++++++++

``EasySemanticChecker`` and ``EasyStatChecker`` are used to check the validity of the learnwares. They are used as:

- ``EasySemanticChecker`` mainly check the integrity and legitimacy of the ``semantic_spec`` in the learnware. A legal ``semantic_spec`` should includes all the keys, and the type of each key should meet our requirements. For keys with type ``Class``, the values should be unique and in our ``valid_list``; for keys with type ``Tag``, the values should not be empty; for keys with type ``String``, a non-empty string is expected as the value; for a table learnware, the dimensions and description of inputs is needed; for ``classification`` or ``regression`` learnwares, the dimensions and description of outputs is indispensable. The learnwares that pass the ``EasySemanticChecker`` is marked as ``NONUSABLE_LEARNWARE``; otherwise, it is ``INVALID_LEARNWARE`` and error information will be returned.
- ``EasyStatChecker`` mainly check the ``model`` and ``stat_spec`` of the learnwares. It includes the following steps:

  - **Check model instantiation**: ``learnware.instantiate_model`` to instantiate the model and transform it to a ``BaseModel``.
  - **Check input shape**: Check whether the shape of ``semantic_spec`` input(if exists), ``learnware.input_shape`` and shape of ``stat_spec`` are consistent, and then generate an example input with that shape. 
  - **Check model prediction**: Use the model to predict the label of the example input, and record the output shape. 
  - **Check output shape**: For ``Classification``, ``Regression`` and ``Feature Extraction`` tasks, the output shape should be consistent with that in ``semantic_spec`` and ``learnware.output_shape``. Besides, for ``Regression`` tasks, the output should be a legal class in ``semantic_spec``.

If any step above fails or meets a error, the learnware will be marked as ``INVALID_LEARNWARE``. The learnwares that pass the ``EasyStatChecker`` is marked as ``USABLE_LEARNWARE``.

``Hetero Market``
--------------------

The learnware market naturally consists of models with different feature spaces, different label spaces, or different objectives. It is beneficial for the market to accommodate these heterogeneous learnwares and provide corresponding learnware recommendation and reuse services to the user so as to expand the applicable scope of learnware paradigm.

Models are submitted to the market with their original specifications. However, these specifications are hard to be used for responding to user requirements due to heterogeneity. Specifications of heterogeneous models reside in different specification spaces. The market needs to merge these specification spaces into a unified one. To achieve this adjustment, you need to implement the class ``EvolvedMarket``, especially the function ``EvolvedMarket.generate_new_stat_specification``, which generates new statistical specifcation in an identical space for each submitted model.

One important case is that models have different feature spaces. In order to enable the learnware market to handle heterogeneous feature spaces, you need to implement the class ``HeterogeneousFeatureMarket`` in the following way:

- First, design a method for the market to connect different feature spaces to a common subspace and implement the function ``HeterogeneousFeatureMarket.learn_mapping_functions``. This function uses specifications of all submitted models to learn mapping functions that can map the data in the original feature space to the common subspace and vice verse.
- Second, use learned mapping functions to implement the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original``.
- Third, use the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original`` to overwrite the mehtod ``EvolvedMarket.generate_new_stat_specification`` and  ``EvolvedMarket.EvolvedMarket.evolve_learnware_list`` of the base class ``EvolvedMarket``.
