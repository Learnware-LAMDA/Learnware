.. _learnware:
==========================================
Learnware & Reuser
==========================================

Learnware and Reuser are related...

Concepts
===================
The learnware paradiam, first introduced by Zhi-Hua Zhou, is defined as a proficiently trained machine learning model accompanied by a specification that allows future users with no prior knowledge of the learnware to identify and reuse it according to their needs.

Developers or owners of trained machine learning models can voluntarily submit their models to a learnware marketplace. If the marketplace accepts the model, it assigns a specification to the model and makes it available in the marketplace.

Utilizing Learnware in Practice
-------------------------------

With a learnware marketplace in place, users can tackle machine learning tasks without having to create models from scratch. 

Addressing Concerns with Learnware
----------------------------------

The learnware approach aims to address several challenges:


+------------------------+----------------------------------------------------------------------------------------+
| Concern                | Solution                                                                               |
+========================+========================================================================================+
| Limited training data  | Use existing high-quality learnware and require only a small amount of data for        |
|                        | adaptation or refinement.                                                              |
+------------------------+----------------------------------------------------------------------------------------+
| Lack of training skills| Leverage existing learnware instead of building a model from scratch.                  |
+------------------------+----------------------------------------------------------------------------------------+
| Catastrophic forgetting| Retain old knowledge in the marketplace as accepted learnware remain available.        |
+------------------------+----------------------------------------------------------------------------------------+
| Continual learning     | Facilitate continuous and lifelong learning with the constant influx of high-quality   |
|                        | learnware, enriching the knowledge base.                                               |
+------------------------+----------------------------------------------------------------------------------------+
| Data privacy and       | Ensure data privacy and proprietary protection by having developers only submit        |
| proprietary concerns   | models, not their data.                                                                |
+------------------------+----------------------------------------------------------------------------------------+
| Unplanned tasks        | Ensure the availability of helpful learnware for various tasks, unless entirely new    |
|                        | to all legal developers.                                                               |
+------------------------+----------------------------------------------------------------------------------------+
| Carbon emissions       | Reduce the need to train numerous large models by assembling smaller models that       |
|                        | provide satisfactory performance.                                                      |
+------------------------+----------------------------------------------------------------------------------------+

Future Work and Progress
------------------------

Despite the promising potential of the learnware proposal, much work remains to bring it to fruition. The following sections will discuss some of the progress made thus far.


Learnware for Hetero Reuse (Feature Aligh + Hetero Map Learnware)
=======================================================================

All Reuse Methods
===========================

JobSelectorReuser
--------------------

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
------------------

The ``AveragingReuser`` is a class that inherits from the base reuse class ``BaseReuser``, that implements the average ensemble method by averaging each learnware's output to predict user data.
There are two parameters required to initialize the class:

- ``learnware_list``: A list of objects of type ``Learnware``.
- ``mode``: The mode of averaging leanrware outputs, which can be set to "mean" or "vote" and defaults to "mean".

If ``mode`` is set to "mean", the ``AveragingReuser`` computes the mean of the learnware's output to predict user data, which is commonly used in regression tasks.
If ``mode`` is set to "vote", the ``AveragingReuser`` computes the mean of the softmax of the learnware's output to predict each label probability of user data, which is commonly used in classification tasks.
