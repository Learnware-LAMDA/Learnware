==========================================
Learnwares Reuse
==========================================

``Learnware Reuser`` is a ``Python API`` that offers a variety of convenient tools for learnware reuse. Users can reuse a single learnware, combination of multiple learnwares,
and heterogeneous learnwares using these tools efficiently, thereby saving the laborious time and effort of building models from scratch. There are mainly two types of 
reuse tools, based on whether user has gathered a small amount of labeled data beforehand: (1) direct reuse and (2) customized reuse based on labeled data.

.. note:: 

    For detailed explanations of the learnware reusers mentioned below, please refer to `COMPONENTS: All Reuse Methods  <../components/learnware.html#all-reuse-methods>`_ .

Homo Reuse
====================

This part introduces baseline methods for reusing homogeneous learnwares to make predictions on unlabeled data.

Direct reuse of Learnware
--------------------------

- ``JobSelector`` selects different learnwares for different data by training a ``job selector`` classifier. The following code shows how to use it:

.. code:: python

    from learnware.reuse import JobSelectorReuser

    # learnware_list is the list of searched learnware
    reuse_job_selector = JobSelectorReuser(learnware_list=learnware_list)

    # test_x is the user's data for prediction
    # predict_y is the prediction result of the reused learnwares
    predict_y = reuse_job_selector.predict(user_data=test_x)

- ``AveragingReuser`` uses an ensemble method to make predictions. The ``mode`` parameter specifies the specific ensemble method:

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


Reusing Learnware with Labeled Data
------------------------------------

When users have a small amount of labeled data, they can also adapt/polish the received learnware(s) by reusing them with the labeled data, gaining even better performance. 

- ``EnsemblePruningReuser`` selectively ensembles a subset of learnwares to choose the ones that are most suitable for the user's task:

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

- ``FeatureAugmentReuser`` helps users reuse learnwares by augmenting features. This reuser regards each received learnware as a feature augmentor, taking its output as a new feature and then build a simple model on the augmented feature set(``logistic regression`` for classification tasks and ``ridge regression`` for regression tasks):

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

When heterogeneous learnware search is activated(see `WORKFLOWS: Hetero Search <../workflows/search.html#hetero-search>`_), users would receive heterogeneous learnwares which are identified from the whole "specification world". 
Though these recommended learnwares are trained from tasks with different feature/label spaces from the user's task, they can still be helpful and perform well beyond their original purpose.
Normally these learnwares are hard to be used, leave alone polished by users, due to the feature/label space heterogeneity. However with the help of ``HeteroMapAlignLearnware`` class which align heterogeneous learnware
with the user's task, users can easily reuse them with the same set of reuse methods mentioned above.

During the alignment process of heterogeneous learnware, the statistical specifications of the learnware and the user's task ``(user_spec)`` are used for input space alignment, 
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
combine ``HeteroMapAlignLearnware`` with the homogeneous reuse methods ``AveragingReuser`` and ``EnsemblePruningReuser`` mentioned above will do the trick:

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

    with LearnwaresContainer(learnware, mode="conda") as env_container: # Let learnware be instance of Learnware Class, and its input shape is (20, 204)
        learnware = env_container.get_learnwares_with_container()[0]
        input_array = np.random.random(size=(20, 204))
        print(learnware.predict(input_array))

The ``mode`` parameter has two options, each for a specific learnware environment loading method:

- ``'conda'``: Install a separate conda virtual environment for each learnware (automatically deleted after execution); run each learnware independently within its virtual environment.
- ``'docker'``: Install a conda virtual environment inside a Docker container (automatically destroyed after execution); run each learnware independently within the container (requires Docker privileges).

.. note:: 
    It's important to note that the "conda" modes are not secure if there are any malicious learnwares. If the user cannot guarantee the security of the learnware they want to load, it's recommended to use the "docker" mode to load the learnware.

