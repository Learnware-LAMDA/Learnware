[![Python Versions](https://img.shields.io/pypi/pyversions/learnware.svg?logo=python&logoColor=white)](https://pypi.org/project/learnware/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/learnware/#files)
[![PypI Versions](https://img.shields.io/pypi/v/learnware)](https://pypi.org/project/learnware/#history)
[![Documentation Status](https://readthedocs.org/projects/learnware/badge/?version=latest)](https://learnware.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/learnware)](LICENSE)


<p align="center">
  <img src="./docs/_static/img/logo/logo1.png" />
</p>

``Learnware Market`` is a model sharing platform, which give a basic implementation of the learnware paradigm. A learnware is a well-performed trained machine learning model with a specification that enables it to be adequately identified to reuse according to the requirement of future users who may know nothing about the learnware in advance. The learnware paradigm can solve entangled problems in the current machine learning paradigm, like continual learning and catastrophic forgetting. It also reduces resources for training a well-performed model.


# Introduction

## Framework 

The learnware paradigm introduces the concept of a well-performed, trained machine learning model with a specification that allows future users, who have no prior knowledge of the learnware, to reuse it based on their requirements.

Developers or owners of trained machine learning models can submit their models to a learnware market. If accepted, the market assigns a specification to the model and accommodates it. The learnware market could host thousands or millions of well-performed models from different developers, for various tasks, using diverse data, and optimizing different objectives.

Instead of building a model from scratch, users can submit their requirements to the learnware market, which then identifies and deploys helpful learnware(s) based on the specifications. Users can apply the learnware directly, adapt it using their data, or exploit it in other ways to improve their model. This process is more efficient and less expensive than building a model from scratch.

## Benefits of the Learnware Paradigm

|  Benefit | Description  |
|  ----  | ----  |
| Lack of training data  | Strong models can be built with small data by adapting well-performed learnwares. |
| Lack of training skills | Ordinary users can obtain strong models by leveraging well-performed learnwares instead of building models from scratch. |
| Catastrophic forgetting  | Accepted learnwares are always stored in the learnware market, retaining old knowledge. |
| Continual learning | The learnware market continually enriches its knowledge with constant submissions of well-performed learnwares. |
| Data privacy/ proprietary | Developers only submit models, not data, preserving data privacy/proprietary. |
| Unplanned tasks | Open to all legal developers, the learnware market can accommodate helpful learnwares for various tasks. |
| Carbon emission | Assembling small models may offer good-enough performance, reducing interest in training large models and the carbon footprint. |

# Quick Start

## Installation

Learnware is currently hosted on [PyPI](https://pypi.org/). You can easily intsall ``Learnware Market`` according to the following steps:

- For Windows and Linux users:

    ```bash
    pip install learnware
    ```

- For macOS users:

    ```bash
    conda install -c pytorch faiss
    pip install learnware
    ```

## Prepare Learnware

The Learnware Market consists of a wide range of learnwares. A valid learnware is a zipfile which 
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

        ```bash
        conda env export | grep -v "^prefix: " > environment.yaml
        ```
        
    - Recover env from config:

        ```bash
        conda env create -f environment.yaml
        ```

We also demonstrate the detail format of learnware zipfile in [DOC link], and also please refer to [Examples](./examples/workflow_by_code/learnware_example) for concrete learnware zipfile example.

## Learnware Market Workflow

Users can start an ``Learnware Market`` workflow according to the following steps:

### Initialize a Learware Market

The ``EasyMarket`` class implements the most basic set of functions in a ``Learnware Market``. 
You can use the following code snippet to initialize a basic ``Learnware Market`` named "demo":

```python
import learnware
from learnware.market import EasyMarket

learnware.init()
easy_market = EasyMarket(market_id="demo", rebuild=True)
```

### Upload Leanwares

Before uploading your learnware into the ``Learnware Market``,
create a semantic specification ``semantic_spec`` by selecting or filling in values for the predefined semantic tags 
to describe the features of your task and model.

For example, the following code snippet demonstrates the semantic specification 
of a Scikit-Learn type model, which is designed for business scenario and performs classification on tabular data:

```python

semantic_spec = {
    "Data": {"Values": ["Tabular"], "Type": "Class"},
    "Task": {"Values": ["Classification"], "Type": "Class"},
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "demo_learnware", "Type": "String"},
}

```

Once the semantic specification is defined, 
you can easily upload your learnware with a single line of code:
    
```python
    
easy_market.add_learnware(zip_path, semantic_spec) 

```

Here, ``zip_path`` is the directory of your learnware zipfile.

### Semantic Specification Search

To search for learnwares that fit your task purpose, 
you should also provide a semantic specification ``user_semantic`` that describes the characteristics of your task.
The ``Learnware Market`` will perform a first-stage search based on ``user_semantic``,
identifying potentially helpful leranwares whose models solve tasks similar to your requirements. 

```python

# construct user_info which includes semantic specification for searching learnware
user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

# search_learnware performs semantic specification search if user_info doesn't include a statistical specification
_, single_learnware_list, _ = easy_market.search_learnware(user_info) 

# single_learnware_list is the learnware list by semantic specification searching
print(single_learnware_list)

```

### Statistical Specification Search

If you choose to porvide your own statistical specification file ``stat.json``, 
the ``Learnware Market`` can perform a more accurate leanware selection from 
the learnwares returned by the previous step. This second-stage search is based on statistical information 
and returns one or more learnwares that are most likely to be helpful for your task. 

For example, the following code is designed to work with Reduced Set Kernel Embedding as a statistical specification:

```python

import learnware.specification as specification

user_spec = specification.rkme.RKMEStatSpecification()
user_spec.load(os.path.join(unzip_path, "rkme.json"))
user_info = BaseUserInfo(
    semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": user_spec}
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

```

### Reuse Learnwares

Based on the returned list of learnwares ``mixture_learnware_list`` in the previous step, 
you can easily reuse them to make predictions your own data, instead of training a model from scratch. 
We provide two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Simply replace ``test_x`` in the code snippet below with your own testing data and start reusing learnwares!

```python

# using jobselector reuser to reuse the searched learnwares to make prediction
reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

# using averaging ensemble reuser to reuse the searched learnwares to make prediction
reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)

```

## Auto Workflow Example

``Learnware Market`` also provides an auto workflow example, which includes preparing learnwares, upload and delete learnware from markets, search learnware with semantic specifications and statistical specifications. The users can run ``examples/workflow_by_code.py`` to try the basic workflow of ``Learnware Market``.


# Experiments and Examples

## Environment

For all experiments, we used a single linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

| System | GPU | CPU | 
| ----  | ----  | ----  |
| Ubuntu 20.04.4 LTS | Nvidia Tesla V100S | Intel(R) Xeon(R) Gold 6240R |



## Datasets

We designed experiments on three publicly available datasets, namely Prediction Future Sales (PFS), M5 Forecasting (M5) and CIFAR 10. For the two sales forecasting data sets of PFS and M5, we divide the user data according to different stores, and train the Ridge model and LightGBM model on the corresponding data respectively. For the CIFAR10 image classification task, we first randomly pick 6 to 10 categories, and randomly select 800 to 2000 samples from each category from the categories corresponding to the training set, constituting a total of 50 different uploaders. For test users, we first randomly pick 3 to 6 categories, and randomly select 150 to 350 samples from each category from the corresponding categories from the test set, constituting a total of 20 different users.

We tested the efficiency of the specification generation and the accuracy of the search and reuse model respectively. The evaluation index on PFS and M5 data is RMSE, and the evaluation index on CIFAR10 classification task is classification accuracy

## Results

The time-consuming specification generation is shown in the table below:

| Dataset | Data Dimensions | Specification Generation Time (s) |
|  ----  | ----  | ----  |
|  PFS  | NAN  | NAN  |
|  M5  | NAN | 9~15  |
|  CIFAR 10 | 9000*3*32*32 | 7~10  |


The accuracy of search and reuse is shown in the table below:

| Dataset | Top-1 Performance | Job Selector Reuse | Average Ensemble Reuse |
|  ----  | ----  | ----  | ----  |
|  PFS  | NAN  | NAN  |  NAN  |
|  M5  | 2.066 +/- 0.424 | 2.116 +/- 0.472  |  2.512 +/- 0.573  |
|  CIFAR 10 | 0.619 +/- 0.138 | 0.585 +/- 0.056  |  .715 +/- 0.075  |

# About

## Contributor
We appreciate all contributions and thank all the contributors!

TODO: Here paste the github API after publishing:

[Pic after publish]()

About us
================

Visit [LAMDA's official website](http://www.lamda.nju.edu.cn/MainPage.ashx),
