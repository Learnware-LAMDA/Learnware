.. _market:

================================
Market
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

Easy Market
-------------
Easy market is a basic realization of the learnware market. It consists of 

- **easy_organizer = EasyOrganizer(market_id=market_id, rebuild=rebuild)**
- **easy_searcher = EasySearcher(organizer=easy_organizer)**
- **easy_checker_list = [EasySemanticChecker(), EasyStatChecker()]**

``EasyOrganizer`` has the following functions:

- **reload_market(self, rebuild=False) -> bool**: Reload the learnware market when server restarted, and return a flag indicating whether the market is reloaded successfully.
- **add_learnware(self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None) -> Tuple[str, int]**: Add a learnware with ``learnware_id``, ``semantic_spec`` and model files in ``zip_path`` into the market. Return the ``learnware_id`` and ``learnwere_status``. The ``learnwere_status`` is set ``check_status`` if it is provided, else ``checker`` will be called to generate the ``learnwere_status``.
- **delete_learnware(self, id: str) -> bool**: Delete the learnware with ``id`` from the market, return a flag of whether the deletion is successfully.
- **update_learnware(self, id: str, zip_path: str = None, semantic_spec: dict = None, check_status: int = None)->int**: Update the learnware's ``zip_path``, ``semantic_spec``, ``check_status``. If None, the corresponding item is not updated. Return a flag indicating whether it passed the ``checker``.
- **get_learnware_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]**: Get one target learnware or a list of target learnwares. Return None if the learnware is not found.
- **get_learnware_zip_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]**: Similar to **get_learnware_by_ids**, but return the zip paths.
- **get_learnware_dir_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]**: Similar to **get_learnware_by_ids**, but return the dir paths.
- **get_learnware_ids(self, top: int = None, check_status: int = None) -> List[str]**: Return the top k(k = ``top``) learnware ids with ``check_status``. If ``top`` is None, return all the matching learnwares; if ``check_status`` is None, any status are allowed.
- **get_learnwares(self, top: int = None, check_status: int = None) -> List[Learnware]**: Similar to **get_learnware_ids**, but return list of learnwares instead of ids.
- **reload_learnware(self, learnware_id: str)**: Reload all the attributes of the learnware with ``learnware_id``.
- **get_learnware_info_from_storage(self, learnware_id: str) -> Dict**: Return learnware zip path and semantic_specification from storage.
- **__len__(self)**: Return the number of learnwares in the market.

``EasySearcher`` consists of ``EasyFuzzsematicSearcher`` and ``EasyStatSearcher``. Detailed introduction is in `WORKFLOWS: Learnwares Search <../workflows/search.html>`_.

``EasySemanticChecker`` and ``EasyStatChecker`` are used to check the validity of the learnwares. They are used as:

- **EasySemanticChecker/EasyStatChecker.__call__(self, learnware)**

``EasySemanticChecker`` mainly check the integrity and legitimacy of the ``semantic_spec`` in the learnware. A legal ``semantic_spec`` should includes all the keys, and the type of each key should meet our requirements. For keys with type ``Class``, the values should be unique and in our ``valid_list``; for keys with type ``Tag``, the values should not be empty; for keys with type ``String``, a non-empty string is expected as the value; for a table learnware, the dimensions and description of inputs is needed; for ``classification`` or ``regression`` learnwares, the dimensions and description of outputs is indispensable. The learnwares that pass the ``EasySemanticChecker`` is marked as ``NONUSABLE_LEARNWARE``; otherwise, it is ``INVALID_LEARNWARE`` and error information will be returned.


``EasyStatChecker`` mainly check the ``model`` and ``stat_spec`` of the learnwares. It includes the following steps:

- **Check model instantiation**: ``learnware.instantiate_model`` to instantiate the model and transform it to a ``BaseModel``.
- **Check input shape**: Check whether the shape of ``semantic_spec`` input(if exists), ``learnware.input_shape`` and shape of ``stat_spec`` are consistent, and then generate an example input with that shape. 
- **Check model prediction**: Use the model to predict the label of the example input, and record the output shape. 
- **Check output shape**: For ``Classification``, ``Regression`` and ``Feature Extraction`` tasks, the output shape should be consistent with that in ``semantic_spec`` and ``learnware.output_shape``. Besides, for ``Regression`` tasks, the output should be a legal class in ``semantic_spec``.

If any step above fails or meets a error, the learnware will be marked as ``INVALID_LEARNWARE``. The learnwares that pass the ``EasyStatChecker`` is marked as ``USABLE_LEARNWARE``.

Hetero Market
--------------

The learnware market naturally consists of models with different feature spaces, different label spaces, or different objectives. It is beneficial for the market to accommodate these heterogeneous learnwares and provide corresponding learnware recommendation and reuse services to the user so as to expand the applicable scope of learnware paradigm.

Models are submitted to the market with their original specifications. However, these specifications are hard to be used for responding to user requirements due to heterogeneity. Specifications of heterogeneous models reside in different specification spaces. The market needs to merge these specification spaces into a unified one. To achieve this adjustment, you need to implement the class ``EvolvedMarket``, especially the function ``EvolvedMarket.generate_new_stat_specification``, which generates new statistical specifcation in an identical space for each submitted model.

One important case is that models have different feature spaces. In order to enable the learnware market to handle heterogeneous feature spaces, you need to implement the class ``HeterogeneousFeatureMarket`` in the following way:

- First, design a method for the market to connect different feature spaces to a common subspace and implement the function ``HeterogeneousFeatureMarket.learn_mapping_functions``. This function uses specifications of all submitted models to learn mapping functions that can map the data in the original feature space to the common subspace and vice verse.
- Second, use learned mapping functions to implement the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original``.
- Third, use the functions ``HeterogeneousFeatureMarket.transform_original_to_subspace`` and ``HeterogeneousFeatureMarket.transform_subspace_to_original`` to overwrite the mehtod ``EvolvedMarket.generate_new_stat_specification`` and  ``EvolvedMarket.EvolvedMarket.evolve_learnware_list`` of the base class ``EvolvedMarket``.
