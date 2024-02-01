==========================================
Learnwares Reuse
==========================================

``Learnware Reuser`` is a core module providing various basic reuse methods for convenient learnware reuse. 
Users can efficiently reuse a single learnware, combination of multiple learnwares,
and heterogeneous learnwares using these methods. 
There are two main categories of reuse methods: (1) data-free reusers which reuse learnwares directly and (2) data-dependent reusers which reuse learnwares with a small amount of labeled data.

.. note:: 

    For detailed explanations of the learnware reusers mentioned below, please refer to `COMPONENTS: All Reuse Methods  <../components/learnware.html#all-reuse-methods>`_ .

Homo Reuse
====================

This part introduces baseline methods for reusing homogeneous learnwares to make predictions on unlabeled data.

The most basic way is to directly use a single learnware.

.. code:: python

    # learnware is a single learnware in the search results
    # test_x is the user's data for prediction
    # predict_y is the prediction result of the reused learnware
    predict_y = learnware.predict(user_data=test_x)

Data-Free Reuser
--------------------------

- ``JobSelector`` selects different learnwares for different data by training a ``job selector`` classifier. The following code shows how to use it:

.. code:: python

    from learnware.reuse import JobSelectorReuser

    # learnware_list is the list of searched learnware
    reuse_job_selector = JobSelectorReuser(learnware_list=learnware_list)

    # test_x is the user's data for prediction
    # predict_y is the prediction result of the reused learnwares
    predict_y = reuse_job_selector.predict(user_data=test_x)

- ``AveragingReuser`` uses an ensemble method to make predictions. The ``mode`` parameter specifies the type of ensemble method:

.. code:: python

    from learnware.reuse import AveragingReuser

    # Regression tasks:
    #   - mode="mean": average the learnware outputs.
    # Classification tasks:
    #   - mode="vote_by_label": majority vote for learnware output labels.
    #   - mode="vote_by_prob": majority vote for learnware output label probabilities.
    
    reuse_ensemble = AveragingReuser(
        learnware_list=learnware_list, mode="vote_by_label"
    )
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)


Data-Dependent Reuser
------------------------------------

When users have minor labeled data, they can also adapt the received learnware(s) by reusing them with the labeled data. 

- ``EnsemblePruningReuser`` selects a subset of suitable learnwares using a multi-objective evolutionary algorithm and uses an average ensemble for prediction:

.. code:: python

    from learnware.reuse import EnsemblePruningReuser

    # mode="regression": Suitable for regression tasks
    # mode="classification": Suitable for classification tasks
    reuse_ensemble_pruning = EnsemblePruningReuser(
        learnware_list=learnware_list, mode="regression"
    )

    # (val_X, val_y) is the small amount of labeled data
    reuse_ensemble_pruning.fit(val_X, val_y)
    predict_y = reuse_job_selector.predict(user_data=test_x)

- ``FeatureAugmentReuser`` assists in reusing learnwares by augmenting features. It concatenates the output of the original learnware with the user's task features, creating enhanced labeled data, on which a simple model is then trained (logistic regression for classification tasks and ridge regression for regression tasks):

.. code:: python

    from learnware.reuse import FeatureAugmentReuser

    # mode="regression": Suitable for regression tasks
    # mode="classification": Suitable for classification tasks
    reuse_feature_augment = FeatureAugmentReuser(
        learnware_list=learnware_list, mode="regression"
    )

    # (val_X, val_y) is the small amount of labeled data
    reuse_feature_augment.fit(val_X, val_y)
    predict_y = reuse_feature_augment.predict(user_data=test_x)


Hetero Reuse
====================

When heterogeneous learnware search is activated, 
users receive potentially helpful heterogeneous learnwares which are identified from the whole "specification world"(see `WORKFLOWS: Hetero Search <../workflows/search.html#hetero-search>`_). 
Normally, these learnwares cannot be directly applied to their tasks due to discrepancies in input and prediction spaces. 
Nevertheless, the ``learnware`` package facilitates the reuse of heterogeneous learnwares through ``HeteroMapAlignLearnware``, 
which aligns the input and output domain of learnwares to match those of the users' tasks. 
These feature-aligned learnwares can then be utilized with either data-free reusers or data-dependent reusers.

During the alignment process of a heterogeneous learnware, the statistical specifications of the learnware and the user's task ``(user_spec)`` are used for input space alignment, 
and a small amount of labeled data ``(val_x, val_y)`` is mandatory to be used for output space alignment. This can be done by the following code:

.. code:: python

    from learnware.reuse import HeteroMapAlignLearnware

    # mode="regression": For user tasks of regression
    # mode="classification": For user tasks of classification
    hetero_learnware = HeteroMapAlignLearnware(learnware=leanrware, mode="regression")
    hetero_learnware.align(user_spec, val_x, val_y)

    # Make predictions using the aligned heterogeneous learnware
    predict_y = hetero_learnware.predict(user_data=test_x)

To reuse multiple heterogeneous learnwares, 
combine ``HeteroMapAlignLearnware`` with the homogeneous reuse methods ``AveragingReuser`` and ``EnsemblePruningReuser`` mentioned above:

.. code:: python

    hetero_learnware_list = []
    for learnware in learnware_list:
        hetero_learnware = HeteroMapAlignLearnware(learnware, mode="regression")
        hetero_learnware.align(user_spec, val_x, val_y)
        hetero_learnware_list.append(hetero_learnware)
                
    # Reuse multiple heterogeneous learnwares using AveragingReuser
    reuse_ensemble = AveragingReuser(learnware_list=hetero_learnware_list, mode="mean")
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)

    # Reuse multiple heterogeneous learnwares using EnsemblePruningReuser
    reuse_ensemble = EnsemblePruningReuser(
        learnware_list=hetero_learnware_list, mode="regression"
    )
    reuse_ensemble.fit(val_x, val_y)
    ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=test_x)

Reuse with ``Model Container``
================================

The ``learnware`` package provides ``Model Container`` to build executive environment for learnwares according to their runtime dependent files. The learnware's model will be executed in the containers and its env will be installed and uninstalled automatically.

Run the following codes to try run a learnware with ``Model Container``:

.. code-block:: python

    from learnware.learnware import Learnware

    # Let learnware be instance of Learnware Class, test_x be an input array
    with LearnwaresContainer(learnware, mode="conda") as env_container: 
        learnware = env_container.get_learnwares_with_container()[0]
        print(learnware.predict(test_x))

The ``mode`` parameter includes two options, each corresponding to a specific learnware environment loading method:

- ``'conda'``: Install a separate conda virtual environment for each learnware (automatically deleted after execution); run each learnware independently within its virtual environment.
- ``'docker'``: Install a conda virtual environment inside a Docker container (automatically destroyed after execution); run each learnware independently within the container (requires Docker privileges).

.. note:: 
    It's important to note that the "conda" modes are not secure if there are any malicious learnwares. If the user cannot guarantee the security of the learnware they want to load, it's recommended to use the "docker" mode to load the learnware.

