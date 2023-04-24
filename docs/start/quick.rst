.. _quick:
============================================================
Quick Start
============================================================


Introduction
==================== 

This ``Quick Start`` guide tries to demonstrate that it's very easy to build a complete Learnware Market workflow and use ``Learnware Market`` to deal with users' tasks.


Installation
====================

Learnware is currently hosted on `PyPI <https://pypi.org/>`__. You can easily intsall ``learnware`` according to the following steps:

- For Windows and Linux users:

    .. code-block::

        pip install learnware

- For macOS users:

    .. code-block::

        conda install -c pytorch fais
        pip install learnware


Prepare Learnware
====================

The Learnware Market consists of a wide range of learnwares. A valid learnware is a zip file which 
is composed of the following four parts.

- ``__init__.py``

    A python file offering interfaces for your model's fitting, predicting and fine-tuning.

- ``rkme.json``

    A json file containing the statistical specification of your data. 

- ``learnware.yaml``
    
    A config file describing your model class name, type of statistical specification(e.g. Reduced Kernel Mean Embedding, ``RKMEStatSpecification``), and 
    the file name of your statistical specification file.

- ``environment.yaml``

    A Conda environment configuration file for running the model (if the model environment is incompatible, you can rely on this for manual configuration). 
    You can generate this file according to the following steps:

    - Create env config for conda:

        .. code-block::

            conda env export | grep -v "^prefix: " > environment.yaml
        
    - Recover env from config:

        .. code-block::

            conda env create -f environment.yaml

We also demonstrate the detail format of learnware zipfile in [DOC link], and also please refer to [Code link] for concrete learnware zipfile example.

Learnware Market Workflow
============================

Users can start an Learnware Market workflow according to the following steps:

Initialize a Learware Market
-------------------------------

The ``EasyMarket`` class implements the most basic set of functions in a Learnware Market. 
You can use the following code snippet to initialize a basic Learnware Market named "demo":

.. code-block:: python
    
    import learnware
    from learnware.market import EasyMarket

    learnware.init()
    easy_market = EasyMarket(market_id="demo", rebuild=True)

Upload Leanwares
-------------------------------

Before uploading your learnware into the Learnware Market,
create a semantic specification ``semantic_spec`` by selecting or filling in values for the predefined semantic tags 
to describe the features of your task and model.

For example, the following code snippet demonstrates the semantic specification 
of a Scikit-Learn type model, which is designed for business scenario and performs classification on tabular data:

.. code-block:: python

    semantic_spec = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "demo_learnware", "Type": "String"},
    }

Once the semantic specification is defined, 
you can easily upload your learnware with a single line of code:
    
.. code-block:: python
    
    easy_market.add_learnware(zip_path, semantic_spec) 

Here, ``zip_path`` is the directory of your learnware zip file.

Semantic Specification Search
-------------------------------

To search for learnwares that fit your task purpose, 
you should also provide a semantic specification ``user_semantic`` that describes the characteristics of your task.
The Learnware Market will perform a first-stage search based on ``user_semantic``,
identifying potentially helpful leranwares whose models solve tasks similar to your requirements. 

.. code-block:: python

    # construct user_info which includes semantic specification for searching learnware
    user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

    # search_learnware performs semantic specification search if user_info doesn't include a statistical specification
    _, single_learnware_list, _ = easy_market.search_learnware(user_info) 

    # single_learnware_list is the learnware list by semantic specification searching
    print(single_learnware_list)
    

Statistical Specification Search
---------------------------------

If you choose to porvide your own statistical specification file ``stat.json``, 
the Learnware Market can perform a more accurate leanware selection from 
the learnwares returned by the previous step. This second-stage search is based on statistical information 
and returns one or more learnwares that are most likely to be helpful for your task. 

For example, the following code is designed to work with Reduced Set Kernel Embedding as a statistical specification:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.rkme.RKMEStatSpecification()
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    user_info = BaseUserInfo(
        id="user", semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": user_spec}
    )
    (sorted_score_list, single_learnware_list,
        mixture_score, mixture_learnware_list) = easy_market.search_learnware(user_info)

    # sorted_score_list is the learnware scores based on MMD distances, sorted in descending order
    print(sorted_score_list) 

    # single_learnware_list is the learnwares sorted in descending order based on their scores
    print(single_learnware_list)

    # mixture_learnware_list is the learnwares whose mixture is helpful for your task
    print(mixture_learnware_list) 

    # mixture_score is the score of the mixture of learnwares
    print(mixture_score)


Reuse Learnwares
-------------------------------

Based on the returned list of learnwares ``mixture_learnware_list`` in the previous step, 
you can easily reuse them to make predictions your own data, instead of training a model from scratch. 
We provide two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Simply replace ``test_x`` in the code snippet below with your own testing data and start reusing learnwares!

.. code-block:: python

    # using jobselector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

    # using averaging ensemble reuser to reuse the searched learnwares to make prediction
    reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)

Auto Workflow Example
============================

``Learnware Market`` also provides an auto workflow example, which includes preparing learnwares, upload and delete learnware from markets, search learnware with semantic specifications and statistical specifications. The users can run ``examples/workflow_by_code.py`` to try the basic workflow of ``Learnware Market``.