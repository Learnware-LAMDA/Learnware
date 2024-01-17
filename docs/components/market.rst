.. _market:

================================
Learnware Market
================================

The ``Learnware Market`` receives high-performance machine learning models from developers, incorporates them into the system, and provides services to users by identifying and reusing learnware to help users solve current tasks. Developers voluntarily submit various learnwares to the learnware market, and the market conducts quality checks and further organization of these learnwares. When users submit task requirements, the learnware market automatically selects whether to recommend a single learnware or a combination of multiple learnwares. 

The ``Learnware Market`` will receive various kinds of learnwares, and learnwares from different feature/label spaces form numerous islands of specifications. All these islands constitute the ``specification world`` in the learnware market. The market should discover and establish connections between different islands and merge them into a unified specification world. This further organization of learnwares supports search learnwares among all learnwares, not just among learnwares that have the same feature space and label space with the user's task requirements.

Framework
======================================

The ``Learnware Market`` is combined with a ``organizer``, a ``searcher``, and a list of ``checker``\ s. 

The ``organizer`` can store and organize learnwares in the market. It supports ``add``, ``delete``, and ``update`` operations for learnwares. It also provides the interface for the ``searcher`` to search learnwares based on user requirements.

The ``searcher`` can search learnwares based on user requirements. The implementation of ``searcher`` depends on the concrete implementation and interface for ``organizer``, where usually an ``organizer`` can be compatible with multiple different ``searcher``\ s.

The ``checker`` is used for checking the learnware in some standards. It should check the utility of a learnware and return the status and a message related to the learnware's check result. Only the learnwares who passed the ``checker`` could be able to be stored and added into the ``Learnware Market``. 



Current Checkers
======================================

The ``learnware`` package provides two different implementations of ``Learnware Market`` where both share the same ``checker`` list. So we first introduce the details of ``checker``\ s.

The ``checker``\ s check a learnware object in different aspects, including environment configuration (``CondaChecker``), semantic specifications (``EasySemanticChecker``), and statistical specifications (``EasyStatChecker``). Each checker's ``__call__`` method is designed to be invoked as a function to conduct the respective checks on the learnware and return the outcomes. It defines three types of learnwares: ``INVALID_LEARNWARE`` denotes the learnware does not pass the check, ``NONUSABLE_LEARNWARE`` denotes the learnware passes the check but cannot make predictions, ``USABLE_LEARNWARE`` denotes the leanrware pass the check and can make predictions. Currently, we have three ``checker``\ s, which are described below.


``CondaChecker``
------------------
This ``checker`` checks the environment of the learnware object. It creates a ``LearnwaresContainer`` instance to handle the Learnware and uses ``inner_checker`` to check the Learnware. If an exception occurs, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status and error message.


``EasySemanticChecker``
-------------------------
This ``checker`` checks the semantic specification of a learnware object. It checks if the given semantic specification conforms to predefined standards. It verifies each key in a predefined dictionary. If the check fails, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status and error message.


``EasyStatChecker``
---------------------

This ``checker`` checks the statistical specification and functionality of a learnware object. It performs multiple checks to validate the learnware. It checks for model instantiation, verifies input shape and statistical specifications, and tests output shape using randomly generated data. In case of exceptions, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status and error message.


Current Markets
======================================

The ``learnware`` package provides two different implementations of ``market``, i.e., ``Easy Market`` and ``Hetero Market``. They have different implementations of ``organizer`` and ``searcher``.

Easy Market
-------------

Easy market is a basic realization of the learnware market. It consists of ``EasyOrganizer``, ``EasySearcher``, and the checker list ``[EasySemanticChecker, EasyStatChecker]``.


``Easy Organizer``
++++++++++++++++++++

``EasyOrganizer`` mainly has the following methods to store learnwares, which is an easy way to organize learnwares.

- **reload_market**: Reload the learnware market when the server restarts and return a flag indicating whether the market is reloaded successfully.
- **add_learnware**: Add a learnware with ``learnware_id``, ``semantic_spec`` and model files in ``zip_path`` into the market. Return the ``learnware_id`` and ``learnwere_status``. The ``learnwere_status`` is set to ``check_status`` if it is provided. Otherwise, the ``checker`` will be called to generate the ``learnwere_status``.
- **delete_learnware**: Delete the learnware with ``id`` from the market and return a flag indicating whether the deletion is successful.
- **update_learnware**: Update the learnware's ``zip_path``, ``semantic_spec``, ``check_status``. If None, the corresponding item is not updated. Return a flag indicating whether it passed the ``checker``.
- **get_learnwares**: Similar to **get_learnware_ids**, but return list of learnwares instead of ids.
- **reload_learnware**: Reload all the attributes of the learnware with ``learnware_id``.

``Easy Searcher``
++++++++++++++++++++

``EasySearcher`` consists of ``EasyFuzzsemanticSearcher`` and ``EasyStatSearcher``. ``EasyFuzzsemanticSearcher`` is a kind of ``Semantic Specification Searcher``, while ``EasyStatSearcher`` is a kind of ``Statistical Specification Searcher``. All these searchers return helpful learnwares based on ``BaseUserInfo`` provided by users.

``BaseUserInfo`` is a ``Python API`` for users to provide enough information to identify helpful learnwares.
When initializing ``BaseUserInfo``, three optional information can be provided: ``id``, ``semantic_spec`` and ``stat_info``. These specifications' introductions are shown in `COMPONENTS: Specification <./spec.html>`_.


The semantic specification search and statistical specification search have been integrated into the same interface ``EasySearcher``. 

- **EasySearcher.__call__(self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**

  - It conducts the semantic searcher ``EasyFuzzsematicSearcher``  on all the learnwares from the ``organizer`` with the same ``check_status`` (All learnwares if ``check_status`` is None). If the result is not empty and the ``stat_info`` is provided in ``user_info``, it conducts ``EasyStatSearcher`` and returns the ``SearchResults``.


``Semantic Specification Searcher``
''''''''''''''''''''''''''''''''''''

``Semantic Specification Searcher`` is the first-stage search based on ``user_semantic``, identifying potentially helpful learnwares whose models solve tasks similar to your requirements. There are two types of Semantic Specification Search: ``EasyExactSemanticSearcher`` and ``EasyFuzzSemanticSearcher``. 

In these two searchers, each learnware in the ``learnware_list`` is compared with ``user_info`` according to their ``semantic_spec`` and added to the search result if matched. Two semantic_spec are matched when all the key words are matched or empty in ``user_info``. Different keys have different matching rules. Their ``__call__`` functions are the same:

- **EasyExactSemanticSearcher/EasyFuzzSemanticSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo)-> SearchResults**

  - For keys ``Data``, ``Task``, ``Library`` and ``license``, two``semantic_spec`` keys are matched only if these values(only one value foreach key) of learnware ``semantic_spec`` exists in values(may be muliplevalues for one key) of user ``semantic_spec``.
  - For the key ``Scenario``, two ``semantic_spec`` keys are matched iftheir values have nonempty intersections.
  - For keys ``Name`` and ``Description``, the values are strings and caseis ignored. In ``EasyExactSemanticSearcher``, two ``semantic_spec`` keys are matched if these values of learnware ``semantic_spec`` is a substring of user ``semantic_spec``. In ``EasyFuzzSemanticSearcher``, it starts with the same kind of exact semantic search as ``EasyExactSemanticSearcher``. If the result is empty, the fuzz semantic searcher is activated:  the ``learnware_list`` is sorted according to the fuzz score function ``fuzzpartial_ratio`` in ``rapidfuzz``.

The results are returned and stored in ``single_results`` of ``SearchResults``.


``Statistical Specification Searcher``
''''''''''''''''''''''''''''''''''''''''''

If the user's statistical specification ``stat_info`` is provided,  the learnware market can perform a more accurate learnware selection using ``EasyStatSearcher``. 

- **EasyStatSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**
 
  - It searches for helpful learnwares from ``learnware_list`` based on the ``stat_info`` in ``user_info``.
  - The result ``SingleSearchItem`` and ``MultipleSearchItem`` are both stored in ``SearchResults``. In ``SingleSearchItem``, it searches for individual learnware solutions for the user's task, and it also assigns scores to indicate the compatibility of each learnware with the user's task. In ``MultipleSearchItem``, it searches for a mixture of learnwares that could solve the user task better; the mixture learnware list and a score for the mixture are returned.
  - The parameter ``search_method`` provides two choice of search strategies for mixture learnwares: ``greedy`` and ``auto``. For the search method ``greedy``, each time it chooses a learnware to make their mixture closer to the user's ``stat_info``; for the search method ``auto``, it directly calculates the best mixture weight for the ``learnware_list``.
  - For single learnware search, we only return the learnwares with a score larger than 0.6. For multiple learnware search, the parameter ``max_search_num`` specifies the maximum length of the returned mixture learnware list. 


``Easy Checker``
++++++++++++++++++++

``EasySemanticChecker`` and ``EasyStatChecker`` are used to check the validity of the learnwares. They are used as:

- ``EasySemanticChecker`` mainly check the integrity and legitimacy of the ``semantic_spec`` in the learnware. A legal ``semantic_spec`` should include all the keys, and the type of each key should meet our requirements. For keys with type ``Class``, the values should be unique and in our ``valid_list``; for keys with type ``Tag``, the values should not be empty; for keys with type ``String``, a non-empty string is expected as the value; for a table learnware, the dimensions and description of inputs are needed; for ``classification`` or ``regression`` learnwares, the dimensions and description of outputs are indispensable. The learnwares that pass the ``EasySemanticChecker`` is marked as ``NONUSABLE_LEARNWARE``; otherwise, it is ``INVALID_LEARNWARE``, and error information will be returned.
- ``EasyStatChecker`` mainly check the ``model`` and ``stat_spec`` of the learnwares. It includes the following steps:

  - **Check model instantiation**: ``learnware.instantiate_model`` to instantiate the model and transform it to a ``BaseModel``.
  - **Check input shape**: Check whether the shape of ``semantic_spec`` input(if it exists), ``learnware.input_shape``, and the shape of ``stat_spec`` are consistent, and then generate an example input with that shape. 
  - **Check model prediction**: Use the model to predict the label of the example input and record the output shape. 
  - **Check output shape**: For ``Classification``, ``Regression`` and ``Feature Extraction`` tasks, the output shape should be consistent with that in ``semantic_spec`` and ``learnware.output_shape``. Besides, for ``Regression`` tasks, the output should be a legal class in ``semantic_spec``.

If any step above fails or meets an error, the learnware will be marked as ``INVALID_LEARNWARE``. The learnwares that pass the ``EasyStatChecker`` are marked as ``USABLE_LEARNWARE``.


Hetero Market
-------------

The Hetero Market encompasses ``HeteroMapTableOrganizer``, ``HeteroSearcher``, and the checker list ``[EasySemanticChecker, EasyStatChecker]``.
It represents an extended version of the Easy Market, capable of accommodating table learnwares from diverse feature spaces (referred to as heterogeneous table learnwares), thereby broadening the applicable scope of the learnware paradigm.
This market trains a heterogeneous engine by utilizing existing learnware specifications to merge distinct specification islands and assign new specifications, referred to as ``HeteroMapTableSpecification``, to learnwares.
As more learnwares are submitted, the heterogeneous engine will undergo continuous updates, with the aim of constructing a more precise specification world.


``HeteroMapTableOrganizer``
+++++++++++++++++++++++++++

``HeteroMapTableOrganizer`` overrides methods from ``EasyOrganizer`` and implements new methods to support the organization of heterogeneous table learnwares. Key features include:

- **reload_market**: Reloads the heterogeneous engine if there is one. Otherwise, initialize an engine with default configurations. Returns a flag indicating whether the market is reloaded successfully.
- **reset**: Resets the heterogeneous market with specific settings regarding the heterogeneous engine such as ``auto_update``, ``auto_update_limit`` and ``training_args`` configurations.
- **add_learnware**: Add a learnware into the market, meanwhile assigning ``HeteroMapTableSpecification`` to the learnware using the heterogeneous engine. The engine's update process will be triggered if ``auto_update`` is set to True and the number of learnwares in the market with ``USABLE_LEARNWARE`` status exceeds ``auto_update_limit``. Return the ``learnware_id`` and ``learnwere_status``.
- **delete_learnware**: Removes the learnware with ``id`` from the market and also removes its new specification if there is one. Return a flag of whether the deletion is successful.
- **update_learnware**: Update the learnware's ``zip_path``, ``semantic_spec``, ``check_status`` and its new specification if there is one. Return a flag indicating whether it passed the ``checker``.
- **generate_hetero_map_spec**: Generate ``HeteroMapTableSpecification`` for users based on the information provided in ``user_info``.
- **train**: Build the heterogeneous engine using learnwares from the market that supports heterogeneous market training.


``HeteroSearcher``
++++++++++++++++++

``HeteroSearcher`` builds upon ``EasySearcher`` with additional support for searching among heterogeneous table learnwares, returning helpful learnwares with feature space and label space different from the user's task requirements.
The semantic specification search and statistical specification search have been integrated into the same interface ``HeteroSearcher``.

- **HeteroSearcher.__call__(self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy") -> SearchResults**

  - It conducts the semantic searcher ``EasyFuzzsematicSearcher``  on all the learnwares from the ``HeteroOrganizer`` with the same ``check_status`` (All learnwares if ``check_status`` is None).
  - If ``stat_info`` is provided within ``user_info``, it conducts one of two types of statistical specification searches using ``EasySearcher``, depending on whether heterogeneous learnware search is enabled. If enabled, ``stat_info`` will be updated with a user-specific ``HeteroMapTableSpecification``, and the Hetero Market performs heterogeneous learnware search based on the updated ``stat_info``. If not enabled, the Hetero Market performs homogeneous learnware search based on the original ``stat_info``.
  
.. note:: 
  The heterogeneous learnware search is enabled when ``user_info`` contains valid heterogeneous search information. Please refer to `WORKFLOWS: Hetero Search  <../workflows/search.html#hetero-search>`_ for details.