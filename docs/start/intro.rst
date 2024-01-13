.. _intro:
================
Introduction
================

The *learnware* paradigm, proposed by Professor Zhi-Hua Zhou in 2016 [1, 2], aims to build a vast model platform system, i.e., a *learnware dock system*, which systematically accommodates and organizes models shared by machine learning developers worldwide, and can efficiently identify and assemble existing helpful model(s) to solve future tasks in a unified way.

The ``learnware`` package provides a fundamental implementation of the central concepts and procedures within the learnware paradigm. Its well-structured design ensures high scalability and facilitates the seamless integration of additional features and techniques in the future.

In addition, the ``learnware`` package serves as the engine for the `Beimingwu System <https://bmwu.cloud/#/>`_ and can be effectively employed for conducting experiments related to learnware.

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

Machine learning has achieved great success in many fields but still faces various challenges, such as the need for extensive training data and advanced training techniques, the difficulty of continuous learning, the risk of catastrophic forgetting, and the leakage of data privacy.

Although there are many efforts focusing on one of these issues separately, they are entangled, and solving one problem may exacerbate others. The learnware paradigm aims to address many of these challenges through a unified framework.

+-----------------------+-----------------------------------------------------------------------------------------------+
| Benefit               | Description                                                                                   |
+=======================+===============================================================================================+
| Lack of training data | Strong models can be built with small data by adapting well-performed learnwares.             |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Lack of training      | Ordinary users can obtain strong models by leveraging well-performed learnwares instead of    |
| skills                | building models from scratch.                                                                 |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Catastrophic          | Accepted learnwares are always stored in the learnware market, retaining old knowledge.       |
| forgetting            |                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Continual learning    | The learnware market continually enriches its knowledge with constant submissions of          |
|                       | well-performed learnwares.                                                                    |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Data privacy/         | Developers only submit models, not data, preserving data privacy/proprietary.                 |
| proprietary           |                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Unplanned tasks       | Open to all legal developers, the learnware market can accommodate helpful learnwares for     |
|                       | various tasks.                                                                                |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Carbon emission       | Assembling small models may offer good-enough performance, reducing interest in training      |
|                       | large models and the carbon footprint.                                                        |
+-----------------------+-----------------------------------------------------------------------------------------------+

How to Solve Future Tasks with Learnware Paradigm?
----------------------------------------------------

.. image:: ../_static/img/learnware_paradigm.jpg
   :align: center

Instead of building a model from scratch, users can submit their requirements to the learnware market, which then identifies and deploys helpful learnware(s) based on the specifications. Users can apply the learnware directly, adapt it using their data, or exploit it in other ways to improve their models. This process is more efficient and less expensive than building a model from scratch.


Procedure of Learnware Paradigm
==================================
- ``Submitting Stage``: Developers voluntarily submit various learnwares to the learnware market, and the system conducts quality checks and further organization of these learnwares.
- ``Deploying Stage``: When users submit task requirements, the learnware market automatically selects whether to recommend a single learnware or a combination of multiple learnwares and provides efficient deployment methods. Whether it's a single learnware or a combination of multiple learnwares, the system offers convenient learnware reuse interfaces.

.. image:: ../_static/img/learnware_market.svg
   :align: center


Framework and Architecture Design
==================================

.. image:: ../_static/img/learnware_framework.svg
   :align: center

The architecture is designed based on the guidelines including *decoupling*, *autonomy*, *reusability*, and *scalability*. The above architecture diagram illustrates the architecture and framework from the perspectives of both modules and workflows.

- At the workflow level, the ``learnware`` package consists of ``Submitting Stage`` and ``Deploying Stage``.

+---------------------+-------------------------------------------------------------------------------------------------------------------+
|      Module         |                                          Workflow                                                                 |
+=====================+===================================================================================================================+
| ``Submitting Stage``| The learnware developers submit learnwares to the learnware market, which conducts usability checks and further   |
|                     | organization of these learnwares.                                                                                 |
+---------------------+-------------------------------------------------------------------------------------------------------------------+
| ``Deploying Stage`` | The `learnware` package identifies learnwares according to users’ task requirements and provides efficient        |
|                     | reuse and deployment methods.                                                                                     |
+---------------------+-------------------------------------------------------------------------------------------------------------------+

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

