<div align=center>
  <img src="./docs/_static/img/logo/logo.svg"  width="420" height="auto" style="max-width: 100%;"/>
  <br/>
  <br/>
</div>

<div align="center">
    <a href="https://pypi.org/project/learnware/#files">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/learnware.svg?logo=python&logoColor=white">
    </a>
    <a href="https://pypi.org/project/learnware/#files">
        <img alt="Platform" src="https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey">
    </a>
    <a href="https://pypi.org/project/learnware/#history">
        <img alt="PypI Versions" src="https://img.shields.io/pypi/v/learnware">
    </a>
    <a href="https://img.shields.io/pypi/dm/example-package">
        <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/example-package">
    </a>
    <a href="https://learnware.readthedocs.io/en/latest/?badge=latest">
        <img alt="Documentation Status" src="https://readthedocs.org/projects/learnware/badge/?version=latest">
    </a>
    <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/pypi/l/learnware">
    </a>
</div>

<div>
    <h3 align="center">
        <a href="README_zh.md">中文</a> |
        <b>English</b>
    </h3>
</div>

# Introduction

_Learnware_ paradigm was proposed by Professor Zhi-Hua Zhou in 2016 [1, 2]. In the _learnware paradigm_, developers worldwide can share models with the _learnware dock system_, which effectively searches for and reuse learnware(s) to help users solve machine learning tasks efficiently without starting from scratch.

The `learnware` package provides a fundamental implementation of the central concepts and procedures within the learnware paradigm. Its well-structured design ensures high scalability and facilitates the seamless integration of additional features and techniques in the future.

In addition, the `learnware` package serves as the engine for the [Beimingwu System](https://bmwu.cloud) and can be effectively employed for conducting experiments related to learnware.

[1] Zhi-Hua Zhou. Learnware: on the future of machine learning. _Frontiers of Computer Science_, 2016, 10(4): 589–590 <br/>
[2] Zhi-Hua Zhou. Machine Learning: Development and Future. _Communications of CCF_, 2017, vol.13, no.1 (2016 CNCC keynote)

## Learnware Paradigm

A learnware consists of a high-performance machine learning model and specifications that characterize the model, i.e., "Learnware = Model + Specification". These specifications, encompassing both semantic and statistical aspects, detail the model's functionality and statistical information, making it easier for future users to identify and reuse these models.

<div align="center">
  <img src="./docs/_static/img/learnware_market.svg" width="700" height="auto" style="max-width: 100%;" />
</div>

The above diagram illustrates the learnware paradigm, which consists of two distinct stages:
- `Submitting Stage`: Developers voluntarily submit various learnwares to the learnware doc system, and the system conducts quality checks and further organization of these learnwares.
- `Deploying Stage`: When users submit task requirements, the learnware doc system automatically selects whether to recommend a single learnware or a combination of multiple learnwares and provides efficient deployment methods. Whether it’s a single learnware or a combination of multiple learnwares, the system offers convenient learnware reuse interfaces.

## Framework and Infrastructure Design 

<div align="center">
  <img src="./docs/_static/img/learnware_framework.svg" width="700" height="auto" style="max-width: 100%;"/>
</div>

The architecture is designed based on the guidelines including _decoupling_, _autonomy_, _reusability_, and _scalability_. The above diagram illustrates the framework from the perspectives of both modules and workflows.

- At the workflow level, the `learnware` package consists of `Submitting Stage` and `Deploying Stage`.

<div align=center>

|  Module | Workflow  |
|  ----  | ----  |
| `Submitting Stage`  | The developers submit learnwares to the learnware market, which conducts usability checks and further organization of these learnwares.  |
| `Deploying Stage` | The learnware market recommends learnwares according to users’ task requirements and provides efficient reuse and deployment methods. |

</div>

- At the module level, the `learnware` package is a platform that consists of `Learnware`, `Market`, `Specification`, `Model`, `Reuse`, and `Interface` modules.

<div align=center>

|  Module | Description  |
|  ----  | ----  |
| `Learnware`  | The specific learnware, consisting of specification module, and user model module. |
| `Market` | Designed for learnware organization, identification, and usability testing. |
| `Specification` | Generating and storing statistical and semantic information of learnware, which can be used for learnware search and reuse. |
| `Model` | Including the base model and the model container, which can provide unified interfaces and automatically create isolated runtime environments. |
| `Reuse` | Including the data-free reuser, data-dependent reuser, and aligner, which can deploy and reuse learnware for user tasks. |
| `Interface` | The interface for network communication with the `Beimingwu` backend.|

</div>



# Quick Start

## Installation

Learnware is currently hosted on [PyPI](https://pypi.org/project/learnware/). You can easily install `learnware` by following these steps:

```bash
pip install learnware
```

In the `learnware` package, besides the base classes, many core functionalities such as "learnware specification generation" and "learnware deployment" rely on the `torch` library. Users have the option to manually install `torch`, or they can directly use the following command to install the `learnware` package:

```bash
pip install learnware[full]
```

**Note:** However, it's crucial to note that due to the potential complexity of the user's local environment, installing `learnware[full]` does not guarantee that `torch` will successfully invoke `CUDA` in the user's local setting.

## Prepare Learnware

In the `learnware` package, each learnware is encapsulated in a `zip` package, which should contain at least the following four files:

- `learnware.yaml`: learnware configuration file.
- `__init__.py`: methods for using the model.
- `stat.json`: the statistical specification of the learnware. Its filename can be customized and recorded in learnware.yaml.
- `environment.yaml` or `requirements.txt`: specifies the environment for the model.

To facilitate the construction of a learnware, we provide a [Learnware Template](https://www.bmwu.cloud/static/learnware-template.zip) that users can use as a basis for building their own learnware. We've also detailed the format of the learnware `zip` package in [Learnware Preparation](https://learnware.readthedocs.io/en/latest/workflows/upload.html#prepare-learnware).

## Learnware Package Workflow

Users can start a `learnware` workflow according to the following steps:

### Initialize a Learnware Market

You can initialize a basic `Learnware Market` named "demo" using the code snippet below:

```python
from learnware.market import instantiate_learnware_market

# instantiate a demo market
demo_market = instantiate_learnware_market(market_id="demo", name="easy", rebuild=True)
```

### Upload Learnware

Before uploading your learnware to the `Learnware Market`, you'll need to create a semantic specification, `semantic_spec`. This involves selecting or inputting values for semantic tags to describe the features of your task and model.

For instance, the following code illustrates the semantic specification for a Scikit-Learn type model. This model is tailored for education scenarios and performs classification tasks on tabular data:

```python
from learnware.specification import generate_semantic_spec

semantic_spec = generate_semantic_spec(
    name="demo_learnware",
    data_type="Table",
    task_type="Classification",
    library_type="Scikit-learn",
    scenarios="Education",
    license="MIT",
)
```

After preparing the semantic specification, you can insert your learnware into the learnware market using a single line of code:

```python
demo_market.add_learnware(zip_path, semantic_spec)
```

Here, `zip_path` is the file path of your learnware `zip` package.

### Semantic Specification Search

To identify learnwares that align with your task's purpose, you'll need to provide a semantic specification, `user_semantic`, that outlines your task's characteristics. The `Learnware Market` will then perform an initial search based on `user_semantic`, which filters learnwares by considering the semantic information of your task.

```python
# construct user_info, which includes a semantic specification
user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

# search_learnware: performs semantic specification search when user_info doesn't include a statistical specification
search_result = demo_market.search_learnware(user_info) 
single_result = search_results.get_single_results()

# single_result: the List of Tuple[Score, Learnware] returned by semantic specification search
print(single_result)
```

### Statistical Specification Search

If you generate and provide a statistical specification file `rkme.json`, the `Learnware Market` will conduct learnware identification based on statistical information, and return more targeted models. Using the API we provided, you can easily generate this statistical specification locally.

For example, the code below executes learnware search when using Reduced Kernel Mean Embedding (RKME) as the statistical specification:
```python
import learnware.specification as specification

user_spec = specification.RKMETableSpecification()

# unzip_path: directory for unzipped learnware zipfile
user_spec.load(os.path.join(unzip_path, "rkme.json"))
user_info = BaseUserInfo(
    semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec}
)
search_result = demo_market.search_learnware(user_info)

single_result = search_results.get_single_results()
multiple_result = search_results.get_multiple_results()

# search_item.score: based on MMD distances, sorted in descending order
# search_item.learnware.id: id of learnwares, sorted by scores in descending order
for search_item in single_result:
    print(f"score: {search_item.score}, learnware_id: {search_item.learnware.id}")

# mixture_item.learnwares: collection of learnwares whose combined use is beneficial
# mixture_item.score: score assigned to the combined set of learnwares in `mixture_item.learnwares`
for mixture_item in multiple_result:
    print(f"mixture_score: {mixture_item.score}\n")
    mixture_id = " ".join([learnware.id for learnware in mixture_item.learnwares])
    print(f"mixture_learnware: {mixture_id}\n")
```

### Reuse Learnwares

We offer two data-free methods `JobSelectorReuser` and `AveragingReuser` for reusing a given list of learnwares. Please substitute ``test_x`` in the code snippet below with your own testing data:

```python
from learnware.reuse import JobSelectorReuser, AveragingReuser

# using job selector reuser to reuse the searched learnwares to make prediction
reuse_job_selector = JobSelectorReuser(learnware_list=mixture_item.learnwares)
job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

# using averaging ensemble reuser to reuse the searched learnwares to make prediction
reuse_ensemble = AveragingReuser(learnware_list=mixture_item.learnwares)
ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
```

We also provide two data-dependent methods: `EnsemblePruningReuser` and `FeatureAugmentReuser`, when the user has minor labeled data for refining a given list of learnwares. Here's an example for adopting multiple returned learnwares by labeled data to solve classification tasks: 

```python
from learnware.reuse import EnsemblePruningReuser, FeatureAugmentReuser

# Use ensemble pruning reuser to reuse the searched learnwares to make prediction
# (train_x, train_y) is the small amount of labeled data
# `mode` has two options "classification" and "regression"
reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_ensemble.fit(train_x, train_y)
ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=test_x)

# Use feature augment reuser to reuse the searched learnwares to make prediction
reuse_feature_augment = FeatureAugmentReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_feature_augment.fit(train_x, train_y)
feature_augment_predict_y = reuse_feature_augment.predict(user_data=test_x)
```

### Auto Workflow Example

The `learnware` package also offers automated workflow examples. This includes preparing learnwares, uploading and deleting learnwares from the market, and searching for learnwares using both semantic and statistical specifications. To experience the basic workflow of the `learnware` package, the users can run `test/test_workflow/test_workflow.py` to try the basic workflow of `learnware`.

# Experiments and Examples

We build various types of experimental scenarios and conduct extensive empirical study to evaluate the baseline algorithms for specification generation, learnware identification, and reuse on tabular, image, and text data.

## Environment

For all experiments, we used a single Linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

<div align=center>

| System               | GPU                | CPU                      |
|----------------------|--------------------|--------------------------|
| Ubuntu 20.04.4 LTS   | Nvidia Tesla V100S | Intel(R) Xeon(R) Gold 6240R |

</div>

## Tabular Scenario Experiments

On various tabular datasets, we initially evaluate the performance of identifying and reusing learnwares from the learnware market that share the same feature space as the user's tasks. Additionally, since tabular tasks often come from heterogeneous feature spaces, we also assess the identification and reuse of learnwares from different feature spaces.

### Settings

Our study utilize three public datasets in the field of sales forecasting: [Predict Future Sales (PFS)](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data), [M5 Forecasting (M5)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), and [Corporacion](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data). To enrich the data, we apply diverse feature engineering methods to these datasets. Then we divide each dataset by store and further split the data for each store into training and test sets. A LightGBM is trained on each Corporacion and PFS training set, while the test sets and M5 datasets are reversed to construct user tasks. This results in an experimental market consisting of 265 learnwares, encompassing five types of feature spaces and two types of label spaces. All these learnwares have been uploaded to the [Beimingwu system](https://bmwu.cloud/).

### Baseline algorithms
The most basic way to reuse a learnware is Top-1 reuser, which directly uses the single learnware chosen by RKME specification. Besides, we implement two data-free reusers and two data-dependent reusers that works on single or multiple helpful learnwares identified from the market. When users have no labeled data, JobSelector reuser selects different learnwares for different samples by training a job selector classifier; AverageEnsemble reuser uses an ensemble method to make predictions. In cases where users possess both test data and limited labeled training data, EnsemblePruning reuser selectively ensembles a subset of learnwares to choose the ones that are most suitable for the user’s task; FeatureAugment reuser regards each received learnware as a feature augmentor, taking its output as a new feature and then builds a simple model on the augmented feature set. JobSelector and FeatureAugment are only effective for tabular data, while others are also useful for text and image data.

### Homogeneous Cases

In the homogeneous cases, the 53 stores within the PFS dataset function as 53 individual users. Each store utilizes its own test data as user data and applies the same feature engineering approach used in the learnware market. These users could subsequently search for homogeneous learnwares within the market that possessed the same feature spaces as their tasks.

We conduct a comparison among different baseline algorithms when the users have no labeled data or limited amounts of labeled data. The average losses over all users are illustrated in the table below. It shows that unlabeled methods are much better than random choosing and deploying one learnware from the market.

<div align=center>

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 0.897  |
| Best in Market (Single)           | 0.756  |
| Top-1 Reuse (Single)              | 0.830  |
| Job Selector Reuse (Multiple)     | 0.848  |
| Average Ensemble Reuse (Multiple) | 0.816  |

</div>

The figure below showcases the results for different amounts of labeled data provided by the user; for each user, we conducted multiple experiments repeatedly and calculated the mean and standard deviation of the losses; the average losses over all users are illustrated in the figure. It illustrates that when users have limited training data, identifying and reusing single or multiple learnwares yields superior performance compared to user's self-trained models. 

<div align=center>
  <img src="./docs/_static/img/Homo_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

### Heterogeneous Cases

Based on the similarity of tasks between the market's learnwares and the users, the heterogeneous cases can be further categorized into different feature engineering and different task scenarios.

#### Different Feature Engineering Scenarios

We consider the 41 stores within the PFS dataset as users, generating their user data using a unique feature engineering approach that differ from the methods employed by the learnwares in the market. As a result, while some learnwares in the market are also designed for the PFS dataset, the feature spaces do not align exactly. 

In this experimental setup, we examine various data-free reusers. The results in the following table indicate that even when users lack labeled data, the market exhibits strong performance, particularly with the AverageEnsemble method that reuses multiple learnwares.

<div align=center>

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 1.149  |
| Best in Market (Single)           | 1.038  |
| Top-1 Reuse (Single)              | 1.075  |
| Average Ensemble Reuse (Multiple) | 1.064  |

</div>


#### Different Task Scenarios

We employ three distinct feature engineering methods on all the ten stores from the M5 dataset, resulting in a total of 30 users. Although the overall task of sales forecasting aligns with the tasks addressed by the learnwares in the market, there are no learnwares specifically designed to satisfy the M5 sales forecasting requirements. 

In the following figure, we present the loss curves for the user's self-trained model and several learnware reuse methods. It is evident that heterogeneous learnwares prove beneficial with a limited amount of the user's labeled data, facilitating better alignment with the user's specific task. 

<div align=center>
  <img src="./docs/_static/img/Hetero_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>


## Image Scenario Experiment

Second, we assess our algorithms on image datasets. It is worth noting that images of different sizes could be standardized through resizing, eliminating the need to consider heterogeneous feature cases.

### Settings

We choose the famous image classification dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60000 32x32 color images in 10 classes. A total of 50 learnwares are uploaded: each learnware contains a convolutional neural network trained on an unbalanced subset that includs 12000 samples from four categories with a sampling ratio of `0.4:0.4:0.1:0.1`. 
A total of 100 user tasks are tested and each user task consists of 3000 samples of CIFAR-10 with six categories with a sampling ratio of `0.3:0.3:0.1:0.1:0.1:0.1`.

### Results

We assess the average performance of various methods using 1 - Accuracy as the loss metric. The following table and figure show that when users face a scarcity of labeled data or possess only a limited amount of it (less than 2000 instances), leveraging the learnware market can yield good performances.

<div align=center>

| Setting                           | Error |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.655    |
| Best in Market (Single)           | 0.304    |
| Top-1 Reuse (Single)              | 0.406    |
| Job Selector Reuse (Multiple)     | 0.406    |
| Average Ensemble Reuse (Multiple) | 0.310    |

</div>


<div align=center>
  <img src="./docs/_static/img/image_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

## Text Scenario Experiment

Finally, we evaluate our algorithms on text datasets. Text data naturally exhibit feature heterogeneity, but this issue can be addressed by applying a sentence embedding extractor.

### Settings

We conduct experiments on the well-known text classification dataset: [20-newsgroup](http://qwone.com/~jason/20Newsgroups/), which consists approximately 20000 newsgroup documents partitioned across 20 different newsgroups. Similar to the image experiments, a total of 50 learnwares are uploaded. Each learnware is trained on a subset that includes only half of the samples from three superclasses and the model in it is a tf-idf feature extractor combined with a naive Bayes classifier. We define 10 user tasks, and each of them encompasses two superclasses.

### Results

The results are depicted in the following table and figure. Similarly, even when no labeled data is provided, the performance achieved through learnware identification and reuse can match that of the best learnware in the market. Additionally, utilizing the learnware market allows for a reduction of approximately 2000 samples compared to training models from scratch.

<div align=center>

| Setting                           | Accuracy |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.507    |
| Best in Market (Single)           | 0.859    |
| Top-1 Reuse (Single)              | 0.846    |
| Job Selector Reuse (Multiple)     | 0.845    |
| Average Ensemble Reuse (Multiple) | 0.862    |

</div>


<div align=center>
  <img src="./docs/_static/img/text_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>


# Citation

If you use our project in your research or work, we kindly request that you cite the following papers:

```bibtex
@article{zhou2024learnware,
  title = {Learnware: Small models do big},
  author = {Zhou, Zhi-Hua and Tan, Zhi-Hao},
  journal = {Science China Information Sciences},
  volume = {67},
  number = {1},
  pages = {112102},
  year = {2024}
}

@inproceedings{tan2024beimingwu,
  title   = {Beimingwu: A Learnware Dock System},
  author  = {Tan, Zhi-Hao and Liu, Jian-Dong and Bi, Xiao-Dong and Tan, Peng and Zheng, Qin-Cheng and Liu, Hai-Tian and Xie, Yi and Zou, Xiao-Chuan and Yu, Yang and Zhou, Zhi-Hua},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  pages  = {5773--5782},
  year    = {2024}
}
```

# About

## How to Contribute

`learnware` is still in its early stages and may contain bugs and issues. We highly value and encourage contributions from the community, and we sincerely invite everyone to research and improve learnware algorithms, develop and explore within the learnware paradigm. For detailed development guidelines, please consult our [Developer Guide](https://learnware.readthedocs.io/en/latest/about/dev.html). We kindly request that contributors adhere to the provided commit format and pre-commit configuration when participating in the project. Your valuable contributions are greatly appreciated.

## About Us

The `learnware` repository is developed and maintained by the LAMDA Beimingwu R&D (Research and Development) Team. To learn more about our team, please visit the [Team Overview](https://docs.bmwu.cloud/en/about-us.html).
