.. _quick:
============================================================
Quick Start
============================================================


Introduction
====================

This ``Quick Start`` guide tries to demonstrate

- It's very easy to build a complete Learnware Market workflow and use ``learnware`` to deal with users' tasks.


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

            conda env export | grep -v "^prefix: " > environment.yml
        
    - Recover env from config:

        .. code-block::

            conda env create -f environment.yml


Learnware Market Workflow
============================

Users can start an Learnware Market workflow according to the following steps:

Initialize a Learware Market
-------------------------------

.. code-block:: python
    
    import learnware
    from learnware.market import EasyMarket

    learnware.init()
    easy_market = EasyMarket(market_id="demo", rebuild=True)

Upload Leanwares
-------------------------------

Here, ``zip_path`` is the directory of your learnware zip file.

.. code-block:: python

    semantic_spec = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "learnware_1", "Type": "String"},
    }
    semantic_spec["Name"]["Values"] = "learnware_user"
    semantic_spec["Description"]["Values"] = "test_learnware_user" 
    easy_market.add_learnware(zip_path, semantic_spec) 

Semantic Specification Search
-------------------------------

The Learnware Market will perform first-step searching based on the semantic specification 
``semantic_spec`` you provided. 
This searching process will indentify potentially helpful leranwares whose models
solve tasks similar to your requirements.

.. code-block:: python

    user_semantic = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Library": {"Values": ["Scikit-learn"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Class"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "", "Type": "String"},
    }
    user_info = BaseUserInfo(id="user", semantic_spec=user_semantic)
    _, single_learnware_list, _ = easy_market.search_learnware(user_info)

Statistical Specification Search
---------------------------------

If you choose to porvide your own statistical specification file ``rkme.json``, 
the Learnware Market can perform a more accurate leanware selection from 
the learnwares returned by the previous step. This second-step searching is carried out 
at the level of data distribution information and returns 
one or more learnwares that are most likely to be helpful for your task.

Here, ``unzip_path`` is the directory where you unzip your learnware file.

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.rkme.RKMEStatSpecification()
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    user_info = BaseUserInfo(
        id="user", semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": user_spec}
    )
    (sorted_score_list, single_learnware_list,
        mixture_score, mixture_learnware_list) = easy_market.search_learnware(user_info)

Reuse Learnwares
-------------------------------

Based on the returned list of learnwares ``mixture_learnware_list`` in the previous step, 
you can easily reuse them to make predictions your own data, instead of training a model from scratch. 
We provide two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.

.. code-block:: python

    reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

    reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode='vote')
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
