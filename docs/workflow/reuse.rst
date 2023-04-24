==========================================
Getting Started-Workflow-Reuse learnware
==========================================

This part introduces two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Instead of training a model from scratch, the user can easily reuse a list of learnwares (``List[Learnware]``) to predict the labels of their own data (``numpy.ndarray`` or ``torch.Tensor``).

To illustrate, we provide a code demonstration that obtains the user dataset using ``sklearn.datasets.load_digits``, where ``test_data`` represents the data that requires prediction.
Assuming that ``learnware_list`` is the list of learnwares searched by the learnware market based on user specifications, the user can reuse each learnware in the ``learnware_list`` through ``JobSelectorReuser`` or ``AveragingReuser`` to predict the label of ``test_data``, thereby avoiding training a model from scratch.

.. code-block:: python

    from sklearn.datasets import load_digits
    from learnware.learnware import JobSelectorReuser, AveragingReuser

    # Load user data
    X, y = load_digits(return_X_y=True)
    test_data = X

    # Based on user information, the learnware market returns a list of learnwares (learnware_list)
    # Use jobselector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_data)

    # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
    reuse_ensemble = AveragingReuser(learnware_list=learnware_list)
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_data)


JobSelectorReuser
====================

The ``JobSelectorReuser`` is a class that inherits from the base reuse class ``BaseReuser``.
Its purpose is to create a job selector that identifies the optimal learnware for each data point in user data.
There are three parameters required to initialize the class:

- ``learnware_list``: A list of objects of type ``Learnware``. Each ``Learnware`` object should have an RKME specification.
- ``herding_num``: An optional integer that specifies the number of items to herd, which defaults to 1000 if not provided.
- ``use_herding``: A boolean flag indicating whether to use kernel herding.

The job selector is essentially a multi-class classifier :math:`g(\boldsymbol{x}):\mathcal{X}\rightarrow \mathcal{I}` with :math:`\mathcal{I}=\{1,\ldots, C\}`, where :math:`C` is the size of ``learnware_list``.
Given a testing sample :math:`\boldsymbol{x}`, the ``JobSelectorReuser`` predicts it by using the :math:`g(\boldsymbol{x})`-th learnware in ``learnware_list``.
If ``use_herding`` is set to false, the ``JobSelectorReuser`` uses data points in each learware's RKME spefication with the corresponding learnware index to train a job selector.
If ``use_herding`` is true, the algorithm estimates the mixture weight based on RKME specifications and raw user data, uses the weight to generate ``herding_num`` auxiliary data points mimicking the user distribution through the kernel herding method, and learns a job selector on these data.


AveragingReuser
====================

The ``AveragingReuser`` is a class that inherits from the base reuse class ``BaseReuser``, that implements the average ensemble method by averaging each learnware's output to predict user data.
There are two parameters required to initialize the class:

- ``learnware_list``: A list of objects of type ``Learnware``.
- ``mode``: The mode of averaging leanrware outputs, which can be set to "mean" or "vote" and defaults to "mean".

If ``mode`` is set to "mean", the ``AveragingReuser`` computes the mean of the learnware's output to predict user data, which is commonly used in regression tasks.
If ``mode`` is set to "vote", the ``AveragingReuser`` computes the mean of the softmax of the learnware's output to predict each label probability of user data, which is commonly used in classification tasks.
