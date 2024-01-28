.. _market:

================================
Learnware Market
================================

The ``Learnware Market``, serving as the implementation of the learnware doc system, receives high-performance machine learning models from developers, incorporates them into the system, and provides services to users by identifying and reusing learnware to help users solve current tasks. Developers voluntarily submit various learnwares to the learnware doc system, and the market conducts quality checks and further organization of these learnwares. When users submit task requirements, the learnware doc system automatically selects whether to recommend a single learnware or a combination of multiple learnwares. 

The ``Learnware Market`` will receive various kinds of learnwares, and learnwares from different feature and prediction spaces form numerous islands of specifications. Collectively, these islands constitute the ``specification world`` in the learnware doc system. The doc system should discover and establish connections between different islands and integrate them into a unified specification world, with the hope of broadening the search scope and preliminarily supporting learnware identification from the entire learnware collection, not just among learnwares that share the same feature and prediction space with the user's task requirements.

Framework
======================================

The ``Learnware Market`` implements the market module which is designed for learnware organization, identification and usability testing. A single market module consists of one ``organizer`` module, one ``searcher`` module, and multiple ``checker`` modules. 

The ``organizer`` module oversees the storage and organization of learnware, supporting operations such as reloading the entire learnware collection and performing insertions, deletions and updates. 

The ``searcher`` module conducts learnware identification based on user information, which encompasses statistical and semantic specifications. It implements several ``searcher``\ s to retrieve learnwares that meet user requirements and recommends them as search results, where each ``searcher`` employs a different search algorithm.

The ``checker`` module is responsible for checking the usability and quality of learnwares by verifying the availability of semantic and statistical specifications and creating a runtime environment to test learnware models based on the model container. The learnwares that pass the ``checker`` module are then inserted and stored by the organizer module, appearing in the ``Learnware Market``. 



Current Checkers
======================================

The ``checker`` module checks a learnware from different aspects using different ``checker``\ s, including environment configuration (``CondaChecker``), semantic specifications (``EasySemanticChecker``), and statistical specifications (``EasyStatChecker``). 
Each checker's ``__call__`` method is designed to be invoked as a function to conduct the respective checks on the learnware and return the outcomes. 
Three types of learnware statuses are defined: ``INVALID_LEARNWARE`` indicates the learnware fails the check, 
``NONUSABLE_LEARNWARE`` indicates the learnware passes the check but is unable to make predictions, ``USABLE_LEARNWARE`` denotes the learnware passes the check and can make predictions. 
Currently, there are three implemented ``checker``\ s within this module, described as follows.


``CondaChecker``
------------------
This ``checker`` checks the environment of the learnware object. It creates a ``LearnwaresContainer`` instance to containerize the learnware and uses ``inner_checker`` to check the Learnware. If an exception occurs, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status with error message.


``EasySemanticChecker``
-------------------------
This ``checker`` checks the semantic specification of a learnware object. It checks if the given semantic specification conforms to predefined standards. It verifies each key in a predefined dictionary. If the check fails, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status and error message.


``EasyStatChecker``
---------------------

This ``checker`` checks the statistical specification and functionality of a learnware object. It performs multiple checks to validate the learnware. It checks for model instantiation, verifies input shape and statistical specifications, and tests output shape using randomly generated data. In case of exceptions, it logs the error and returns the ``NONUSABLE_LEARNWARE`` status and error message.


Current Markets
======================================

The ``learnware`` package provides two different implementations of ``market``, i.e., ``Easy Market`` and ``Hetero Market``. 
They share the same ``checker`` module and have different implementations of ``organizer`` and ``searcher``.

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

``BaseUserInfo`` is a ``Python API`` for users to provide enough information to identify helpful learnwares. When initializing ``BaseUserInfo``, three optional information can be provided: ``id``, ``semantic_spec`` and ``stat_info``. These specifications' introductions are shown in `COMPONENTS: Specification <./spec.html>`_.


The semantic specification search and statistical specification search have been integrated into the same interface ``EasySearcher``. 

- **EasySearcher.__call__(self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**

  - It conducts the semantic searcher ``EasyFuzzsematicSearcher``  on all the learnwares from the ``organizer`` with the same ``check_status`` (All learnwares if ``check_status`` is None). If the result is not empty and the ``stat_info`` is provided in ``user_info``, it conducts ``EasyStatSearcher`` and returns the ``SearchResults``.


``Semantic Specification Searcher``
''''''''''''''''''''''''''''''''''''

``Semantic Specification Searcher`` is the first-stage search based on ``user_semantic``, identifying potentially helpful learnwares whose models solve tasks similar to your requirements. There are two types of Semantic Specification Search: ``EasyExactSemanticSearcher`` and ``EasyFuzzSemanticSearcher``. 

In these two searchers, each learnware in the ``learnware_list`` is compared with ``user_info`` based on their ``semantic_spec``. A learnware is added to the search result if a match is found. Two ``semantic_spec``\ s are considered matched when all the key words either match or are empty in ``user_info``. Different keys follow different matching rules. The ``__call__`` function for these searchers are the same:

- **EasyExactSemanticSearcher/EasyFuzzSemanticSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo)-> SearchResults**

  - For the keys ``Data``, ``Task``, ``Library``, and ``license`` in ``semantic_spec``, a match occurs only when the value (only one value for each key) in a learnware's ``semantic_spec`` is also found in the values (which may be multiple for one key) in the user's ``semantic_spec``.
  - For the key ``Scenario``, two ``semantic_spec`` keys are matched if their values have nonempty intersections.
  - For the keys ``Name`` and ``Description``, the values are strings and case sensitivity is ignored. In ``EasyExactSemanticSearcher``, two ``semantic_spec`` keys are matched if these values in the learnware ``semantic_spec`` is a substring of the corresponding values in the user ``semantic_spec``. ``EasyFuzzSemanticSearcher`` begins with the same exact semantic search as ``EasyExactSemanticSearcher``. If no results are found, it activates a fuzz semantic searcher:  the ``learnware_list`` is then sorted according to the fuzz score function ``fuzzpartial_ratio`` provided by ``rapidfuzz``.

The results are returned and stored in ``single_results`` of ``SearchResults``.


``Statistical Specification Searcher``
''''''''''''''''''''''''''''''''''''''''''

If the user's statistical specification ``stat_info`` is provided,  the learnware doc system can perform more targeted learnware identification using ``EasyStatSearcher``. 

- **EasyStatSearcher.__call__(self, learnware_list: List[Learnware], user_info: BaseUserInfo, max_search_num: int = 5, search_method: str = "greedy",) -> SearchResults**
 
  - It searches for helpful learnwares from ``learnware_list`` based on the ``stat_info`` in ``user_info``.
  - ``SingleSearchItem`` and ``MultipleSearchItem`` are types of results stored in ``SearchResults`. ``SingleSearchItem``` contains single recommended learnwares for the user's task, along with scores indicating each learnware's compatibility with the task. ``MultipleSearchItem`` includes a combination of learnwares, attempting to address the task better, and provides an overall score for this mixture.
  - The parameter ``search_method`` offers two options for search strategies of mixture learnwares: ``greedy`` and ``auto``. With the ``greedy`` method, it incrementally adds learnwares that significantly reduce the distribution distance, thereby bringing the mixture closer te the user's ``stat_info``. With the the search method ``auto``, it directly calculates the optimal mixture weights for the ``learnware_list``.
  - For single learnware search, only learnwares with a score higher than 0.6 are returned. For multiple learnware search, the parameter ``max_search_num`` specifies the maximum number of learnwares in the returned mixture learnware list. 


``Easy Checker``
++++++++++++++++++++

``EasySemanticChecker`` and ``EasyStatChecker`` are used to verify the validity of the learnwares:

- ``EasySemanticChecker`` checks the integrity and legitimacy of the ``semantic_spec`` in learnware. (1) A valid ``semantic_spec`` must include all necessary keys, with each key's type conforming to specified requirements. For ``Class`` type keys, values should be unique and in the ``valid_list``; for ``Tag`` type keys, values should not be empty; for ``String`` type keys, a non-empty string is expected. (2) Tabular learnwares should include input dimensions and feature descriptions within their ``semantic_spec``; (3) ``Classification`` or ``Regression`` learnwares should provide output dimensions and descriptions. Learnwares passing the ``EasySemanticChecker`` are marked as ``NONUSABLE_LEARNWARE``; otherwise, as ``INVALID_LEARNWARE``, with error information returned.
- ``EasyStatChecker`` checks the ``model`` and ``stat_spec`` of the learnwares, involving:

  - **Model instantiation check**: Utilizing ``learnware.instantiate_model`` to instantiate the model as a ``BaseModel``.
  - **Input shape check**: Checking whether the ``semantic_spec`` input shape (if present), ``learnware.input_shape``, and ``stat_spec`` shape are consistent, and then generating an example input of that shape. 
  - **Model prediction check**: Using the model to predict the label of the example input and recording the model output.
  - **Output shape check**: For ``Classification``, ``Regression``, and ``Feature Extraction`` tasks, the output's shape should align with ``semantic_spec`` and ``learnware.output_shape``. For ``Regression`` tasks, the output's shape should also be consistent with the output dimension provided in the ``semantic_spec``. For ``Classification`` tasks, the output should either contain valid classification labels or match the output dimension provided in the ``semantic_spec``.

If any step above fails or meets an error, the learnware will be marked as ``INVALID_LEARNWARE``. The learnwares that pass the ``EasyStatChecker`` are marked as ``USABLE_LEARNWARE``.


Hetero Market
-------------

The Hetero Market encompasses ``HeteroMapTableOrganizer``, ``HeteroSearcher``, and the checker list ``[EasySemanticChecker, EasyStatChecker]``.
It represents an preliminary extension of the Easy Market, designed to support tabular tasks, with the aim of accommodating tabular learnwares from diverse feature spaces (referred to as heterogeneous table learnwares), 
This extension thereby broadens the search scope and facilitates learnware identification and reuse across the entire learnware selection.
The Hetero Market utilizes existing learnware specifications to train a heterogeneous engine, which merges distinct specification islands and assigns new specifications, known as ``HeteroMapTableSpecification``, to learnwares. 
As more learnwares are submitted, this heterogeneous engine will continuously update, hopefully leading to a more precise specification world.


``HeteroMapTableOrganizer``
+++++++++++++++++++++++++++

``HeteroMapTableOrganizer`` overrides methods from ``EasyOrganizer`` and implements new methods to support the management of heterogeneous table learnwares. Key features include:

- **reload_market**: Reloads the heterogeneous engine if there is one. Otherwise, initialize an engine with default configurations. Returns a flag indicating whether the market is reloaded successfully.
- **reset**: Resets the heterogeneous market with specific settings regarding the heterogeneous engine such as ``auto_update``, ``auto_update_limit`` and ``training_args`` configurations.
- **add_learnware**: Add a learnware into the market, meanwhile generating ``HeteroMapTableSpecification`` for the learnware using the heterogeneous engine. The engine's update process will be triggered if ``auto_update`` is set to True and the number of learnwares in the market with ``USABLE_LEARNWARE`` status exceeds ``auto_update_limit``. Return the ``learnware_id`` and ``learnwere_status``.
- **delete_learnware**: Removes the learnware with ``id`` from the market and also removes its new specification if there is one. Return a flag of whether the deletion is successful.
- **update_learnware**: Update the learnware's ``zip_path``, ``semantic_spec``, ``check_status`` and its new specification if there is one. Return a flag indicating whether it passed the ``checker``.
- **generate_hetero_map_spec**: Generate ``HeteroMapTableSpecification`` for users based on the user's statistical specification provided in ``user_info``.
- **train**: Build the heterogeneous engine using learnwares from the market that supports heterogeneous market training.


``HeteroSearcher``
++++++++++++++++++

``HeteroSearcher`` builds upon ``EasySearcher`` with additional support for searching among heterogeneous table learnwares, returning potentially helpful learnwares with feature and prediction spaces different from the user's task requirements.
The semantic specification search and statistical specification search have been integrated into the same interface ``HeteroSearcher``.

- **HeteroSearcher.__call__(self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy") -> SearchResults**

  - It conducts the semantic searcher ``EasyFuzzsematicSearcher``  on all the learnwares from the ``HeteroOrganizer`` with the same ``check_status`` (All learnwares if ``check_status`` is None).
  - If ``stat_info`` is provided within ``user_info``, it conducts one of two types of statistical specification searches using ``EasySearcher``, depending on whether heterogeneous learnware search is enabled. If enabled, ``stat_info`` will be updated with a user-specific ``HeteroMapTableSpecification``, and the Hetero Market performs heterogeneous learnware search based on the updated ``stat_info``. If not enabled, the Hetero Market performs homogeneous learnware search based on the original ``stat_info``.
  
.. note:: 
  The heterogeneous learnware search is enabled when ``user_info`` contains valid heterogeneous search information. Please refer to `WORKFLOWS: Hetero Search  <../workflows/search.html#hetero-search>`_ for details.