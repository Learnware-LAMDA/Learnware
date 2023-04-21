===========
Quick Start
===========

Introduction
============

This ``Quick Start`` guide tries to demonstrate

- It's very easy to build a complete Learnware Market workflow and use ``Learnware`` to deal with users' tasks.
- Though with public data and simple models, machine learning technologies work very well in practical Quant investment.



Installation
============

Learnware is currently hosted on `PyPI <https://pypi.org/>`__. You can easily intsall ``Learnware`` according to the following steps:

- For Windows and Linux users:

    .. code-block::

        pip install learnware

- For macOS users:

    .. code-block::

        conda install -c pytorch fais
        pip install learnware


Prepare Learnware
============

The Learnware Market consists of a vast amount of learnwares. A valid learnware is composed of four parts. Please refer to
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

    - create env config for conda:

        .. code_block::

            conda env export | grep -v "^prefix: " > environment.yml
        
    - recover env from config:

        .. code_block::

            conda env create -f environment.yml


Auto Quant Research Workflow
============================

``Qlib`` provides a tool named ``qrun`` to run the whole workflow automatically (including building dataset, training models, backtest and evaluation). Users can start an auto quant research workflow and have a graphical reports analysis according to the following steps:

- Quant Research Workflow:
    - Run  ``qrun`` with a config file of the LightGBM model `workflow_config_lightgbm.yaml` as following.

        .. code-block::

            cd examples  # Avoid running program under the directory contains `qlib`
            qrun benchmarks/LightGBM/workflow_config_lightgbm.yaml


    - Workflow result
        The result of ``qrun`` is as follows, which is also the typical result of ``Forecast model(alpha)``. Please refer to  `Intraday Trading <../component/backtest.html>`_. for more details about the result.

        .. code-block:: python

                                                              risk
            excess_return_without_cost mean               0.000605
                                       std                0.005481
                                       annualized_return  0.152373
                                       information_ratio  1.751319
                                       max_drawdown      -0.059055
            excess_return_with_cost    mean               0.000410
                                       std                0.005478
                                       annualized_return  0.103265
                                       information_ratio  1.187411
                                       max_drawdown      -0.075024


    To know more about `workflow` and `qrun`, please refer to `Workflow: Workflow Management <../component/workflow.html>`_.

- Graphical Reports Analysis:
    - Run ``examples/workflow_by_code.ipynb`` with jupyter notebook
        Users can have portfolio analysis or prediction score (model prediction) analysis by run ``examples/workflow_by_code.ipynb``.
    - Graphical Reports
        Users can get graphical reports about the analysis, please refer to `Analysis: Evaluation & Results Analysis <../component/report.html>`_ for more details.



Custom Model Integration
========================

``Qlib`` provides a batch of models (such as ``lightGBM`` and ``MLP`` models) as examples of ``Forecast Model``. In addition to the default model, users can integrate their own custom models into ``Qlib``. If users are interested in the custom model, please refer to `Custom Model Integration <../start/integration.html>`_.
