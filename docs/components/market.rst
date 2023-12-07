.. _market:

================================
Market
================================


Concepts
======================================

In the learnware paradigm, there are three key players: *developers*, *users*, and the *market*. 

* Developers: Typically machine learning experts who create and aim to share or sell their high-performing trained machine learning models. 
* Users: Require machine learning services but often possess limited data and lack the necessary knowledge and expertise in machine learning. 
* Market: Acquires top-performing trained models from developers, houses them within the marketplace, and offers services to users by identifying and reusing learnwares to help users with their current tasks. 

This process can be broken down into two main stages.

Submitting Stage
------------------------------

During the *submitting stage*, developers can voluntarily submit their trained models to the learnware market. The market will then implement a quality assurance mechanism, such as performance validation, to determine if a submitted model is suitable for acceptance. In a learnware market with millions of models, identifying potentially helpful models for a new user is a challenge.

Requiring users to submit their own data to the market for model testing is impractical, time-consuming, and costly, as it could lead to data leakage. Straightforward approaches, such as measuring the similarity between user data and the original training data of models, are also infeasible due to privacy and proprietary concerns. Our design operates under the constraint that the learnware market has no access to the original training data from developers or users. Furthermore, it assumes that users have limited knowledge of the models available in the market.

The solution's crux lies in the *specification*, which is central to the learnware proposal. Once a submitted model is accepted by the learnware market, it is assigned a specification, which conveys the model's specialty and utility without revealing its original training data. For simplicity, consider models as functions that map input domain :math:`\mathcal{X}` to output domain :math:`\mathcal{Y}` with respect to objective 'obj.' These models exist in a functional space :math:`\mathcal{F}: \mathcal{X} \mapsto \mathcal{Y}` with respect to 'obj.' Each model has a specification, and all specifications form a specification space, where those for models that serve similar tasks are situated closely.

In a learnware market, heterogeneous models may have different :math:`\mathcal{X}`, :math:`\mathcal{Y}`, or objectives. If we refer to the specification space that covers all possible models in all possible functional spaces as the 'specification world' analogously, then each specification space corresponding to one possible functional space can be called a 'specification island.' Designing an elegant specification format that encompasses the entire specification world and allows all possible models to be efficiently and adequately identified is a significant challenge. Currently, we adopt a practical design, where each learnware's specification consists of two parts.


Reusing Stage
------------------------------

Creating Learnware Specifications
++++++++++++++++++++++++++++++++++++

The first part of the learnware specification can be realized by a string consisting of a set of descriptions/tags given by the learnware market. These tags address aspects such as the task, input, output, and objective. Based on the user's provided descriptions/tags, the corresponding specification island can be efficiently and accurately located. The designer of the learnware market can create an initial set of descriptions/tags, which can grow as new models are accepted, and new functional spaces and specification islands are created.

Merging Specification Islands
+++++++++++++++++++++++++++++++++

Specification islands can merge into larger ones. For example, when a new model about :math:`F: \mathcal{X}_1 \cup \mathcal{X}_2 \mapsto \mathcal{Y}` with respect to 'obj' is accepted by the learnware market, two islands can be merged. This is possible because the market can have synthetic data by randomly generating inputs, feeding them to models, and concatenating each input with its corresponding output to construct a dataset reflecting the function of a model. In principle, specification islands can be merged if there are common ingredients in :math:`\mathcal{X}`, :math:`\mathcal{Y}`, and 'obj.'

Deploying Learnware Models
++++++++++++++++++++++++++++++

In the deploying stage, the user submits their requirement to the learnware market, which then identifies and returns helpful learnwares to the user. There are two issues to address: how to identify learnwares matching the user requirement and how to reuse the returned learnwares.

The learnware market can house thousands or millions of models. Efficiently identifying helpful learnwares is challenging, especially given that the market has no access to the original training data of learnwares or the current user's data. With the specification design mentioned earlier, the market can request users to describe their intentions using a set of descriptions/tags, through a user interface or a learnware description language. Based on this information, the task becomes identifying helpful learnwares in a specification island.

Reusing Learnwares
++++++++++++++++++++++

Once helpful learnwares are identified and delivered to the user, they can be reused in various ways. Users can apply the received learnware directly to their data, use multiple learnwares to create an ensemble, or adapt and polish the received learnware(s) using their own data. Learnwares can also be used as feature augmentors, with their outputs used as augmented features for building the final model.

Helpful learnwares may be trained for tasks that are not exactly the same as the user's current task. In such cases, users can tackle their tasks in a divide-and-conquer way or reuse the learnwares collectively through measuring the utility of each model on each testing instance. If users find it difficult to express their requirements accurately, they can adapt and polish the received learnwares directly using their own data.


Framework
======================================


Current Markets
======================================

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

Current Checkers
======================================
