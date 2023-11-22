.. _quick:
============================================================
Quick Start
============================================================


Introduction
==================== 

This ``Quick Start`` guide aims to illustrate the straightforward process of establishing a full ``Learnware Market`` workflow 
and utilizing ``Learnware Market`` to handle user tasks.


Installation
====================

Learnware is currently hosted on `PyPI <https://pypi.org/>`__. You can easily intsall ``learnware`` by following these steps:

- For Windows and Linux users:

    .. code-block::

        pip install learnware

- For macOS users:

    .. code-block::

        conda install -c pytorch faiss
        pip install learnware


Prepare Learnware
====================

The Learnware Market encompasses a board variety of learnwares. A valid learnware is a zipfile that
includes the following four components:

- ``__init__.py``

    A Python file that provides interfaces for fitting, predicting, and fine-tuning your model.

- ``rkme.json``

    A JSON file that contains the statistical specification of your data. 

- ``learnware.yaml``
    
    A configuration file that details your model's class name, the type of statistical specification(e.g. ``RKMETableSpecification`` for Reduced Kernel Mean Embedding), and 
    the file name of your statistical specification file.

- ``environment.yaml`` or ``requirements.txt``

    - ``environment.yaml`` for conda:

        A Conda environment configuration file for running the model. If the model environment is incompatible, this file can be used for manual configuration. 
        Here's how you can generate this file:

        - Create env config for conda:

            - For Windows users:
            
                .. code-block::

                    conda env export | findstr /v "^prefix: " > environment.yaml
            
            - For macOS and Linux users

                .. code-block::

                    conda env export | grep -v "^prefix: " > environment.yaml
            
        - Recover env from config:

            .. code-block::

                conda env create -f environment.yaml
    
    - ``requirements.txt`` for pip:

        A plain text documents that lists all packages necessary for executing the model. These dependencies can be effortlessly installed using pip with the command:

            .. code-block::
            
                pip install -r requirements.txt.

We've also detailed the format of the learnware zipfile in :ref:`Learnware Preparation<workflow/submit:Prepare Learnware>`.


Learnware Market Workflow
============================

Users can start a ``Learnware Market`` workflow according to the following steps:

Initialize a Learware Market
-------------------------------

The ``EasyMarket`` class provides the core functions of a ``Learnware Market``. 
You can initialize a basic ``Learnware Market`` named "demo" using the code snippet below:

.. code-block:: python
    
    import learnware
    from learnware.market import EasyMarket

    learnware.init()
    easy_market = EasyMarket(market_id="demo", rebuild=True)


Upload Leanware
-------------------------------

Before uploading your learnware to the ``Learnware Market``, 
you'll need to create a semantic specification, ``semantic_spec``. This involves selecting or inputting values for predefined semantic tags 
to describe the features of your task and model.

For instance, the dictionary snippet below illustrates the semantic specification for a Scikit-Learn type model. 
This model is tailored for business scenarios and performs classification tasks on tabular data:

.. code-block:: python

    semantic_spec = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "demo_learnware", "Type": "String"},
    }

After defining the semantic specification, 
you can upload your learnware using a single line of code:
    
.. code-block:: python
    
    easy_market.add_learnware(zip_path, semantic_spec) 

Here, ``zip_path`` is the directory of your learnware zipfile.


Semantic Specification Search
-------------------------------

To find learnwares that align with your task's purpose, you'll need to provide a semantic specification, ``user_semantic``, that outlines your task's characteristics. 
The ``Learnware Market`` will then perform an initial search using ``user_semantic``, identifying potentially useful learnwares with models that solve tasks similar to your requirements.

.. code-block:: python

    # construct user_info which includes a semantic specification
    user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

    # search_learnware: performs semantic specification search when user_info doesn't include a statistical specification
    _, single_learnware_list, _ = easy_market.search_learnware(user_info) 

    # single_learnware_list: the learnware list returned by semantic specification search
    print(single_learnware_list)
    

Statistical Specification Search
---------------------------------

If you decide in favor of porviding your own statistical specification file, ``stat.json``, 
the ``Learnware Market`` can further refine the selection of learnwares from the previous step. 
This second-stage search leverages statistical information to identify one or more learnwares that are most likely to be beneficial for your task. 

For example, the code below executes learnware search when using Reduced Set Kernel Embedding as the statistical specification:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.RKMETableSpecification()

    # unzip_path: directory for unzipped learnware zipfile
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec}
    )
    (sorted_score_list, single_learnware_list,
        mixture_score, mixture_learnware_list) = easy_market.search_learnware(user_info)

    # sorted_score_list: learnware scores(based on MMD distances), sorted in descending order
    print(sorted_score_list) 

    # single_learnware_list: learnwares, sorted by scores in descending order
    print(single_learnware_list)

    # mixture_learnware_list: collection of learnwares whose combined use is beneficial
    print(mixture_learnware_list) 

    # mixture_score: score assigned to the combined set of learnwares in `mixture_learnware_list`
    print(mixture_score)


Reuse Learnwares
-------------------------------

With the list of learnwares, ``mixture_learnware_list``, returned from the previous step, you can readily apply them to make predictions on your own data, bypassing the need to train a model from scratch. 
We offer two baseline methods for reusing a given list of learnwares: ``JobSelectorReuser`` and ``AveragingReuser``. 
Just substitute ``test_x`` in the code snippet below with your own testing data, and you're all set to reuse learnwares!

.. code-block:: python

    # using jobselector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

    # using averaging ensemble reuser to reuse the searched learnwares to make prediction
    reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)


Auto Workflow Example
============================

The ``Learnware Market`` also offers an automated workflow example. 
This includes preparing learnwares, uploading and deleting learnwares from the market, and searching for learnwares using both semantic and statistical specifications. 
To experience the basic workflow of the Learnware Market, users can run [workflow code link].
