.. _intro:
================
Introduction
================

*Learnware* was proposed by Professor Zhi-Hua Zhou in 2016 [1, 2]. In the *learnware paradigm*, developers worldwide can share models with the *learnware dock system*, which effectively searches for and reuse learnware(s) to help users solve machine learning tasks efficiently without starting from scratch.

The ``learnware`` package provides a fundamental implementation of the central concepts and procedures and encompasses all processes within the *learnware paradigm*, including the submitting, usability testing, organization, identification, deployment and reuse of learnwares. Its well-structured design ensures high scalability and facilitates the seamless integration of additional features and techniques in the future.

In addition, the ``learnware`` package serves as the core engine for the `Beimingwu System <https://bmwu.cloud>`_, which supports the computational and algorithmic aspects of ``Beimingwu`` and offers rich algorithmic interfaces for learnware-related tasks and research experiments.

| [1] Zhi-Hua Zhou. Learnware: on the future of machine learning. *Frontiers of Computer Science*, 2016, 10(4): 589–590
| [2] Zhi-Hua Zhou. Machine Learning: Development and Future. *Communications of CCF*, 2017, vol.13, no.1 (2016 CNCC keynote)

What is Learnware?
====================

A learnware consists of a high-performance machine learning model and specifications that characterize the model, i.e., "Learnware = Model + Specification".

The learnware specification consists of "semantic specification" and "statistical specification":

- ``Semantic Specification``: Describe the type and functionality of the model through text.
- ``Statistical Specification``: Characterize the statistical information contained in the model using various machine learning techniques.

Learnware specifications describe the model's capabilities, enabling the model to be identified and reused by future users who may know nothing about the learnware in advance.

Why do we need Learnware?
============================

The Benefits of Learnware Paradigm
-------------------------------------

Machine learning has achieved great success in many fields but still faces various challenges, such as the need for extensive training data and advanced training techniques, the difficulty of continuous learning, the risk of catastrophic forgetting, and the risk of data privacy breach.

Although many efforts focus on one of these issues separately, these efforts pay less attention to the fact that most issues are entangled in practice. The learnware paradigm aims to tackle many of these challenges through a unified framework:

+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|      Challenges       |                                                                          Learnware Paradigm Solutions                                                                          |
+=======================+================================================================================================================================================================================+
| Lack of training data | Strong models can be built with a small amount of data by refining well-performing learnwares.                                                                                 |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Lack of training      | Users across all levels of expertise can adequately utilize numerous high-quality and potentially helpful learnwares                                                           |
| skills                | identified by the system for their specific tasks.                                                                                                                             |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Catastrophic          | Learnwares which pass the usability checks are always stored in the learnware doc system, retaining old knowledge.                                                             |
| forgetting            |                                                                                                                                                                                |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Continual learning    | The learnware doc system continually expands its knowledge base with constant submissions of                                                                                   |
|                       | well-performed learnwares.                                                                                                                                                     |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data privacy/         | Developers worldwide freely share their high-performing models, without revealing their training data.                                                                         |
| proprietary           |                                                                                                                                                                                |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Unplanned tasks       | Open to all legal developers, the learnware doc system  accommodate helpful learnwares for                                                                                     |
|                       | various tasks, especially for unplanned, specialized, data-sensitive scenarios.                                                                                                |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Carbon emission       | By assembling the most suitable small learnwares, local deployment becomes feasible, offering a practical alternative to large cloud-based models and their carbon footprints. |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

How to Solve Future Tasks with Learnware Paradigm?
----------------------------------------------------

.. image:: ../_static/img/learnware_paradigm.jpg
   :align: center

When a user is going to solve a new machine learning task, she can submit her requirements to the learnware doc system, and then the system will identify and assemble some helpful learnware(s) from numerous learnwares to return to the user based on the learnware specifications. She can apply the learnware(s) directly, adapt them by her own data, or exploit it in other ways to improve her own model. No matter which learnware reuse mechanism is adopted, the whole process can be much less expensive and more efficient than building a model from scratch by herself.


Procedure of Learnware Paradigm
==================================
- ``Submitting Stage``: Developers voluntarily submit various learnwares to the learnware doc system, and the system conducts quality checks and further organization of these learnwares.
- ``Deploying Stage``: The user submits her task requirement to the learnware doc system, and the system will identify and return some helpful learnwares to the user based on specifications, which can be further reused on user data.

.. image:: ../_static/img/learnware_market.svg
   :align: center


Framework and Architecture Design
==================================

.. image:: ../_static/img/learnware_framework.svg
   :align: center

The architecture is designed based on the guidelines including *decoupling*, *autonomy*, *reusability*, and *scalability*. The above architecture diagram illustrates the architecture and framework from the perspectives of both modules and workflows.

- At the workflow level, the ``learnware`` package consists of ``Submitting Stage`` and ``Deploying Stage``.

+----------------------+---------------------------------------------------------------------------------------------------------------------+
|        Module        |                                                      Workflow                                                       |
+======================+=====================================================================================================================+
| ``Submitting Stage`` | The learnware developers submit learnwares to the learnware doc system, which conducts usability checks and further |
|                      | organization of these learnwares.                                                                                   |
+----------------------+---------------------------------------------------------------------------------------------------------------------+
| ``Deploying Stage``  | The `learnware` package identifies learnwares according to users’ task requirements and provides efficient          |
|                      | reuse and deployment methods.                                                                                       |
+----------------------+---------------------------------------------------------------------------------------------------------------------+

- At the module level, the ``learnware`` package is a platform that consists of ``Learnware``, ``Market``, ``Specification``, ``Model``, ``Reuse``, and ``Interface`` modules.

+------------------+------------------------------------------------------------------------------------------------------------+
|      Module      |                                      Description                                                           |
+==================+============================================================================================================+
| ``Learnware``    | The specific learnware, consisting of specification module, and user model module.                         |
+------------------+------------------------------------------------------------------------------------------------------------+
| ``Market``       | Designed for learnware organization, identification, and usability testing.                                |
+------------------+------------------------------------------------------------------------------------------------------------+
| ``Specification``| Generating and storing statistical and semantic information of learnware, which can be used for learnware  |
|                  | search and reuse.                                                                                          |
+------------------+------------------------------------------------------------------------------------------------------------+
| ``Model``        | Including the base model and the model container, which can provide unified interfaces and automatically   |
|                  | create isolated runtime environments.                                                                      |
+------------------+------------------------------------------------------------------------------------------------------------+
| ``Reuse``        | Including the data-free reuser, data-dependent reuser, and aligner, which can deploy and reuse learnware   |
|                  | for user tasks.                                                                                            |
+------------------+------------------------------------------------------------------------------------------------------------+
| ``Interface``    | The interface for network communication with the `Beimingwu` backend.                                      |
+------------------+------------------------------------------------------------------------------------------------------------+

