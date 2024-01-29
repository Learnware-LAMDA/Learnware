.. _quick:
============================================================
Quick Start
============================================================


Introduction
==================== 

This ``Quick Start`` guide aims to illustrate the straightforward process of establishing a full ``Learnware`` workflow and utilizing ``Learnware`` to handle user tasks.


Installation
====================

Learnware is currently hosted on `PyPI <https://pypi.org/project/learnware/>`_. You can easily intsall the ``learnware`` package by following these steps:

.. code-block:: bash

    pip install learnware

In the ``learnware`` package, besides the base classes, many core functionalities such as "learnware specification generation" and "learnware deployment" rely on the ``torch`` library. Users have the option to manually install ``torch``, or they can directly use the following command to install the ``learnware`` package:

.. code-block:: bash

    pip install learnware[full]

.. note:: 
    However, it's crucial to note that due to the potential complexity of the user's local environment, installing ``learnware[full]`` does not guarantee that ``torch`` will successfully invoke ``CUDA`` in the user's local setting.

Prepare Learnware
====================

In the ``learnware`` package, each learnware is encapsulated in a ``zip`` package, which should contain at least the following four files:

- ``learnware.yaml``: learnware configuration file.
- ``__init__.py``: methods for using the model.
- ``stat.json``: the statistical specification of the learnware. Its filename can be customized and recorded in learnware.yaml.
- ``environment.yaml`` or ``requirements.txt``: specifies the environment for the model.

To facilitate the construction of a learnware, we provide a `Learnware Template <https://www.bmwu.cloud/static/learnware-template.zip>`_ that the users can use as a basis for building your own learnware.  We've also detailed the format of the learnware ``zip`` package in `Learnware Preparation <../workflows/upload:prepare-learnware>`_.

Learnware Package Workflow
============================

Users can start a ``Learnware`` workflow according to the following steps:

Initialize a Learnware Market
------------------------------

You can initialize a basic ``Learnware Market`` named "demo" using the code snippet below:

.. code-block:: python
    
    from learnware.market import instantiate_learnware_market

    # instantiate a demo market
    demo_market = instantiate_learnware_market(market_id="demo", name="easy", rebuild=True) 


Upload Leanware
-------------------------------

Before uploading your learnware to the ``Learnware Market``, you'll need to create a semantic specification, ``semantic_spec``. This involves selecting or inputting values for semantic tags to describe the features of your task and model.

For instance, the following code illustrates the semantic specification for a Scikit-Learn type model. This model is tailored for education scenarios and performs classification tasks on tabular data:

.. code-block:: python

    from learnware.specification import generate_semantic_spec

    semantic_spec = generate_semantic_spec(
        name="demo_learnware",
        data_type="Table",
        task_type="Classification",
        library_type="Scikit-learn",
        scenarios="Education",
        license="MIT",
    )

After preparing the semantic specification, you can insert your learnware into the ``Learnware Market`` using a single line of code:
    
.. code-block:: python

    demo_market.add_learnware(zip_path, semantic_spec) 

Here, ``zip_path`` is the directory of your learnware ``zip`` package.


Semantic Specification Search
-------------------------------

To identify learnwares that align with your task's purpose, you'll need to provide a semantic specification, ``user_semantic``, that outlines your task's characteristics. The ``Learnware Market`` will then perform an initial search based on ``user_semantic``, which filters learnwares by considering the semantic information of your task.

.. code-block:: python

    # construct user_info which includes a semantic specification
    user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

    # search_learnware: performs semantic specification search when user_info doesn't include a statistical specification
    search_result = demo_market.search_learnware(user_info) 
    single_result = search_results.get_single_results()

    # single_result: the List of Tuple[Score, Learnware] returned by semantic specification search
    print(single_result)
    

Statistical Specification Search
---------------------------------

If you generate and provide a statistical specification file ``rkme.json``, the ``Learnware Market`` will conduct learnware identification based on statistical information, and return more targeted models. Using the API we provided, you can easily generate this statistical specification locally.

For example, the code below executes learnware search when using Reduced Kernel Mean Embedding (RKME) as the statistical specification:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.RKMETableSpecification()

    # unzip_path: directory for unzipped learnware zipfile
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec}
    )
    search_result = demo_market.search_learnware(user_info)

    single_result = search_results.get_single_results()
    multiple_result = search_results.get_multiple_results()

    # search_item.score: based on MMD distances, sorted in descending order
    # search_item.learnware.id: id of learnwares, sorted by scores in descending order
    for search_item in single_result:
        print(f"score: {search_item.score}, learnware_id: {search_item.learnware.id}")

    # mixture_item.learnwares: collection of learnwares whose combined use is beneficial
    # mixture_item.score: score assigned to the combined set of learnwares in `mixture_item.learnwares`
    for mixture_item in multiple_result:
        print(f"mixture_score: {mixture_item.score}\n")
        mixture_id = " ".join([learnware.id for learnware in mixture_item.learnwares])
        print(f"mixture_learnware: {mixture_id}\n")


Reuse Learnwares
-------------------------------

We offer two data-free methods ``JobSelectorReuser`` and ``AveragingReuser`` for reusing a given list of learnwares. Please substitute ``test_x`` in the code snippet below with your own testing data:

.. code-block:: python

    from learnware.reuse import JobSelectorReuser, AveragingReuser

    # Use job selector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=mixture_item.learnwares)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

    # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
    reuse_ensemble = AveragingReuser(learnware_list=mixture_item.learnwares)
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)


We also provide two data-dependent methods: ``EnsemblePruningReuser`` and ``FeatureAugmentReuser``, when the user has minor labeled data for refining a given list of learnwares. Here's an example for adopting multiple returned learnwares by labeled data to solve classification tasks: 

.. code-block:: python

    from learnware.reuse import EnsemblePruningReuser, FeatureAugmentReuser

    # Use ensemble pruning reuser to reuse the searched learnwares to make prediction
    # (train_x, train_y) is the small amount of labeled data
    # `mode` has two options "classification" and "regression"
    reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_item.learnwares, mode="classification")
    reuse_ensemble.fit(train_X, train_y)
    ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=data_X)

    # Use feature augment reuser to reuse the searched learnwares to make prediction
    reuse_feature_augment = FeatureAugmentReuser(learnware_list=mixture_item.learnwares, mode="classification")
    reuse_feature_augment.fit(train_X, train_y)
    feature_augment_predict_y = reuse_feature_augment.predict(user_data=data_X)

Auto Workflow Example
============================

The `learnware` package also offers automated workflow examples. This includes preparing learnwares, uploading and deleting learnwares from the market, and searching for learnwares using both semantic and statistical specifications. 
To experience the basic workflow of the `learnware` package, the users can run `test/test_workflow/test_workflow.py` to try the basic workflow of `learnware`.
