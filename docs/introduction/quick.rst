===========
Quick Start
===========

Introduction
============

This ``Quick Start`` guide tries to demonstrate

- It's very easy to build a complete Learnware Market workflow and use ``learnware`` to deal with users' tasks.


Installation
============

Learnware is currently hosted on `PyPI <https://pypi.org/>`__. You can easily intsall ``learnware`` according to the following steps:

- For Windows and Linux users:

    .. code-block::

        pip install learnware

- For macOS users:

    .. code-block::

        conda install -c pytorch fais
        pip install learnware


Prepare Learnware
============

The Learnware Market consists of a wide range of learnwares. A valid learnware is a zip file which 
is composed of the following four parts. Please refer to
:ref:`script` for examples of these components.

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

1. Initialize a Learware Market:

    .. code-block:: python
        
        import learnware
        from learnware.market import EasyMarket

        learnware.init()
        easy_market = EasyMarket(market_id="demo", rebuild=True)

2. Upload leanware:

    Here, ``zip_path`` is the directory of your learnware zip file.

    .. code-block:: python

        semantic_spec = {
            "Data": {"Values": ["Tabular"], "Type": "Class"},
            "Task": {"Values": ["Classification"], "Type": "Class"},
            "Device": {"Values": ["GPU"], "Type": "Tag"},
            "Scenario": {"Values": ["Business"], "Type": "Tag"},
            "Description": {"Values": "", "Type": "String"},
            "Name": {"Values": "learnware_1", "Type": "String"},
        }
        semantic_spec["Name"]["Values"] = "learnware_user"
        semantic_spec["Description"]["Values"] = "test_learnware_user" 
        easy_market.add_learnware(zip_path, semantic_spec) 

3. Semantic specification search:

    .. code-block:: python

        user_semantic = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "", "Type": "String"},
    }
    user_info = BaseUserInfo(id="user_0", semantic_spec=user_semantic)
    _, single_learnware_list, _ = easy_market.search_learnware(user_info)

4. Statistical specification search:

5. Reuse learnwares:

.. _script:

Example: Learnware Files
-------

Below is an example learnware that includes an SVM model and uses Reduced Kernel Mean Embedding as its statistical reduction method. 
We have listed the files that it needs to include.