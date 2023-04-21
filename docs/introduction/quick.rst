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

2. Upload leanware:

3. Semantic specification search:

4. Statistical specification search:

5. Reuse learnwares: