==========================================
Getting Started-Workflow-Reuse learnware
==========================================

This part introduces two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Instead of training a model from scratch, you can easily reuse a list of learnwares ``learnware_list (List[Learnware])`` to make predictions on your own data ``test_data (numpy.ndarray or torch.Tensor)`` in the following way:

.. code-block:: python

    from learnware.learnware import JobSelectorReuser, AveragingReuser

    # using jobselector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_data)

    # using averaging ensemble reuser to reuse the searched learnwares to make prediction
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

The job selector is essentially a multi-class classifier :math:`g(\boldsymbol{x}\rightarrow \mathcal{I})` with :math:`\mathcal{I}=\{1,\ldots, C\}`.
Given a testing sample :math:`\boldsymbol{x}`, the ``JobSelectorReuser`` predicts it by using the :math:`g(\boldsymbol{x})`-th learnware in ``learnware_list``.
If ``use_herding`` is set to false, the ``JobSelectorReuser`` uses data points in each learware's RKME spefication with the corresponding learnware index to train a job selector.
If ``use_herding`` is true, the algorithm estimates the mixture weight based on RKME specifications and raw user data, uses the weight to generate ``herding_num`` auxiliary data points mimicking the user distribution through the kernel herding method, and learns a job selector on these data.


AveragingReuser
====================

