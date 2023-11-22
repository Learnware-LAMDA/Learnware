==========================================
Learnwares Reuse
==========================================

This part introduces two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Instead of training a model from scratch, the user can easily reuse a list of learnwares (``List[Learnware]``) to predict the labels of their own data (``numpy.ndarray`` or ``torch.Tensor``).

To illustrate, we provide a code demonstration that obtains the user dataset using ``sklearn.datasets.load_digits``, where ``test_data`` represents the data that requires prediction.
Assuming that ``learnware_list`` is the list of learnwares searched by the learnware market based on user specifications, the user can reuse each learnware in the ``learnware_list`` through ``JobSelectorReuser`` or ``AveragingReuser`` to predict the label of ``test_data``, thereby avoiding training a model from scratch.

Homo Reuse
====================

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

Hetero Reuse
====================


Reuse with Container
=====================
