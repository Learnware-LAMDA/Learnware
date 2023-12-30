[![Python Versions](https://img.shields.io/pypi/pyversions/learnware.svg?logo=python&logoColor=white)](https://pypi.org/project/learnware/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/learnware/#files)
[![PypI Versions](https://img.shields.io/pypi/v/learnware)](https://pypi.org/project/learnware/#history)
[![Documentation Status](https://readthedocs.org/projects/learnware/badge/?version=latest)](https://learnware.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/learnware)](LICENSE)



<div align=center>
  <img src="./docs/_static/img/logo/logo1.png"  width="50%"/>
</div>


``Learnware`` is a model sharing platform, which give a basic implementation of the learnware paradigm. A learnware is a well-performed trained machine learning model with a specification that enables it to be adequately identified to reuse according to the requirement of future users who may know nothing about the learnware in advance. The learnware paradigm can solve entangled problems in the current machine learning paradigm, like continual learning and catastrophic forgetting. It also reduces resources for training a well-performed model.


# Introduction

## Framework 

<div align="center">
  <img src="./docs/_static/img/learnware_paradigm.jpg" width="70%"/>
</div>

Machine learning, especially the prevailing big model paradigm, has achieved great success in natural language processing and computer vision applications. However, it still faces challenges such as the requirement of a large amount of labeled training data, difficulty in adapting to changing environments, and catastrophic forgetting when refining trained models incrementally. These big models, while useful in their targeted tasks, often fail to address the above issues and struggle to generalize beyond their specific purposes.

<div align="center">
  <img src="./docs/_static/img/learnware_market.svg" width="70%" />
</div>

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

## Installation

Learnware is currently hosted on [PyPI](https://pypi.org/). You can easily install `Learnware` by following these steps:

```bash
pip install learnware
```

In the `Learnware` package, besides the base classes, many core functionalities such as "learnware specification generation" and "learnware deployment" rely on the `torch` library. Users have the option to manually install `torch`, or they can directly use the following command to install the `learnware` package:

```bash
pip install learnware[full]
```

**Note:** However, it's crucial to note that due to the potential complexity of the user's local environment, installing `learnware[full]` does not guarantee that `torch` will successfully invoke `CUDA` in the user's local setting.

## Prepare Learnware

In Learnware, each learnware is encapsulated in a `zip` package, which should contain at least the following four files:

- `learnware.yaml`: learnware configuration file.
- `__init__.py`: methods for using the model.
- `stat.json`: the statistical specification of the learnware. Its filename can be customized and recorded in learnware.yaml.
- `environment.yaml` or `requirements.txt`: specifies the environment for the model.

To facilitate the construction of a learnware, we provide a [Learnware Template](https://www.bmwu.cloud/static/learnware-template.zip) that users can use as a basis for building their own learnware. We've also detailed the format of the learnware `zip` package in [Learnware Preparation](docs/workflows/upload:prepare-learnware).

## Learnware Package Workflow

Users can start a `Learnware` workflow according to the following steps:

### Initialize a Learnware Market

The `EasyMarket` class provides the core functions of a `Learnware Market`. You can initialize a basic `Learnware Market` named "demo" using the code snippet below:

```python
from learnware.market import instantiate_learnware_market

# instantiate a demo market
demo_market = instantiate_learnware_market(market_id="demo", name="easy", rebuild=True)
```

### Upload Learnware

Before uploading your learnware to the `Learnware Market`, you'll need to create a semantic specification, `semantic_spec`. This involves selecting or inputting values for predefined semantic tags to describe the features of your task and model.

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

After defining the semantic specification, you can upload your learnware using a single line of code:

```python
demo_market.add_learnware(zip_path, semantic_spec)
```

Here, `zip_path` is the directory of your learnware `zip` package.

### Semantic Specification Search

To find learnwares that align with your task's purpose, you'll need to provide a semantic specification, `user_semantic`, that outlines your task's characteristics. The `Learnware Market` will then perform an initial search using `user_semantic`, identifying potentially useful learnwares with models that solve tasks similar to your requirements.

```python
# construct user_info, which includes a semantic specification
user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

# search_learnware: performs semantic specification search when user_info doesn't include a statistical specification
search_result = easy_market.search_learnware(user_info) 
single_result = search_results.get_single_results()

# single_result: the List of Tuple[Score, Learnware] returned by semantic specification search
print(single_result)
```

### Statistical Specification Search

If you decide in favor of providing your own statistical specification file, `stat.json`, the `Learnware Market` can further refine the selection of learnwares from the previous step. This second-stage search leverages statistical information to identify one or more learnwares that are most likely to be beneficial for your task.

For example, the code below executes learnware search when using Reduced Set Kernel Embedding as the statistical specification:

```python
import learnware.specification as specification

user_spec = specification.RKMETableSpecification()

# unzip_path: directory for unzipped learnware zipfile
user_spec.load(os.path.join(unzip_path, "rkme.json"))
user_info = BaseUserInfo(
    semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec}
)
search_result = easy_market.search_learnware(user_info)

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

With the list of learnwares, `mixture_learnware_list`, returned from the previous step, you can readily apply them to make predictions on your own data, bypassing the need to train a model from scratch. We provide two methods for reusing a given list of learnwares: `JobSelectorReuser` and `AveragingReuser`. Substitute `test_x` in the code snippet below with your testing data, and you're all set to reuse learnwares:

```python
from learnware.reuse import JobSelectorReuser, AveragingReuser

# using jobselector reuser to reuse the searched learnwares to make prediction
reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

# using averaging ensemble reuser to reuse the searched learnwares to make prediction
reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
```

We also provide two methods when the user has labeled data for reusing a given list of learnwares: `EnsemblePruningReuser` and `FeatureAugmentReuser`. Substitute `test_x` in the code snippet below with your testing data, and substitute `train_X, train_y` with your training labeled data, and you're all set to reuse learnwares:

```python
from learnware.reuse import EnsemblePruningReuser, FeatureAugmentReuser

# Use ensemble pruning reuser to reuse the searched learnwares to make prediction
reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_ensemble.fit(train

_X, train_y)
ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=data_X)

# Use feature augment reuser to reuse the searched learnwares to make prediction
reuse_feature_augment = FeatureAugmentReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_feature_augment.fit(train_X, train_y)
feature_augment_predict_y = reuse_feature_augment.predict(user_data=data_X)
```

### Auto Workflow Example

The `Learnware` also offers automated workflow examples. This includes preparing learnwares, uploading and deleting learnwares from the market, and searching for learnwares using both semantic and statistical specifications. To experience the basic workflow of the `Learnware` package, the users can run `test/test_workflow/test_workflow.py` to try the basic workflow of `Learnware`.

# Experiments and Examples

This chapter will introduce related experiments to illustrate the search and reuse performance of our learnware system.

## Environment

For all experiments, we used a single Linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

| System               | GPU                | CPU                      |
|----------------------|--------------------|--------------------------|
| Ubuntu 20.04.4 LTS   | Nvidia Tesla V100S | Intel(R) Xeon(R) Gold 6240R |

## Tabular Data Experiments

### Datasets

Our study involved three public datasets in the sales forecasting field: [Predict Future Sales (PFS)](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data), [M5 Forecasting (M5)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), and [Corporacion](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data).

We applied various pre-processing methods to these datasets to enhance the richness of the data. After pre-processing, we first divided each dataset by store and then split the data for each store into training and test sets. Specifically:

- For PFS, the test set consisted of the last month's data from each store.
- For M5, we designated the final 28 days' data from each store as the test set.
- For Corporacion, the test set was composed of the last 16 days of data from each store.

In the submitting stage, the Corporacion dataset's 55 stores are regarded as 165 uploaders, each employing one of three different feature engineering methods. For the PFS dataset, 100 uploaders are established, each using one of two feature engineering approaches. These uploaders then utilize their respective stores' training data to develop LightGBM models. As a result, the learnware market comprises 265 learnwares, derived from five types of feature spaces and two types of label spaces.

Based on the specific design of user tasks, our experiments were primarily categorized into two types:

- **homogeneous experiments** are designed to evaluate performance when users can reuse learnwares in the learnware market that have the same feature space as their tasks (homogeneous learnwares). This contributes to showing the effectiveness of using learnwares that align closely with the user's specific requirements.

- **heterogeneous experiments** aim to evaluate the performance of identifying and reusing helpful heterogeneous learnwares in situations where no available learnwares match the feature space of the user's task. This helps to highlight the potential of learnwares for applications beyond their original purpose.

### Homogeneous Tabular Dataset

For homogeneous experiments, the 55 stores in the Corporacion dataset act as 55 users, each applying one feature engineering method, and using the test data from their respective store as user data. These users can then search for homogeneous learnwares in the market with the same feature spaces as their tasks.

The Mean Squared Error (MSE) of search and reuse across all users is presented in the table below:

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 0.331  |
| Best in Market (Single)           | 0.151  |
| Top-1 Reuse (Single)              | 0.280  |
| Job Selector Reuse (Multiple)     | 0.274  |
| Average Ensemble Reuse (Multiple) | 0.267  |

When users have both test data and limited training data derived from their original data, reusing single or multiple searched learnwares from the market can often yield better results than training models from scratch on limited training data. We present the change curves in MSE for the user's self-trained model, as well as for the Feature Augmentation single learnware reuse method and the Ensemble Pruning multiple learnware reuse method. These curves display their performance on the user's test data as the amount of labeled training data increases. The average results across 55 users are depicted in the figure below:

<div align=center>
  <img src="./docs/_static/img/table_homo_labeled.png"  width="50%"/>
</div>

From the figure, it's evident that when users have limited training data, the performance of reusing single/multiple table learnwares is superior to that of the user's own model. This emphasizes the benefit of learnware reuse in significantly reducing the need for extensive training data and achieving enhanced results when available user training data is limited.

### Heterogeneous Tabular Dataset

In heterogeneous experiments, the learnware market would recommend helpful heterogeneous learnwares with different feature spaces with the user tasks. Based on whether there are learnwares in the market that handle tasks similar to the user's task, the experiments can be further subdivided into the following two types:

#### Cross Feature Space Experiments

We designate the 41 stores in the PFS dataset as users, creating their user data with an alternative feature engineering approach that varies from the methods employed by learnwares in the market. Consequently, while the market's learnwares from the PFS dataset undertake tasks very similar to our users, the feature spaces do not match exactly. In this experimental configuration, we tested various heterogeneous learnware reuse methods (without using user's labeled data) and compared them to the user's self-trained model based on a small amount of training data. The average MSE performance across 41 users is as follows:

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 1.459  |
| Best in Market (Single)           | 1.226  |
| Top-1 Reuse (Single)              | 1.407  |
| Average Ensemble Reuse (Multiple) | 1.312  |
| User model with 50 labeled data   | 1.267  |

From the results, it is noticeable that the learnware market still performs quite well even when users lack labeled data, provided it includes learnwares addressing tasks that are similar but not identical to the user's. In these instances, the market's effectiveness can match or even rival scenarios where users have access to a limited quantity of labeled data.

#### Cross Task Experiments

Here we have chosen the 10 stores from the M5 dataset to act as users. Although the broad task of sales forecasting is similar to the tasks addressed by the learnwares in the market, there are no learnwares available that directly cater to the M5 sales forecasting requirements. All learnwares show variations in both feature and label spaces compared to the tasks of M5 users. We present the change curves in RMSE for the user's self-trained model and several learnware reuse methods. These curves display their performance on the user's test data as the amount of labeled training data increases. The average results across 10 users are depicted in the figure below:

<div align=center>
  <img src="./docs/_static/img/table_hetero_labeled.png"  width="50%"/>
</div>

We can observe that heterogeneous learnwares are beneficial when there's a limited amount of the user's labeled training data available, aiding in better alignment with the user's specific task. This underscores the potential of learnwares to be applied to tasks beyond their original purpose.

## Image Data Experiment

For the CIFAR-10 dataset, we sampled the training set uneven

ly by category and constructed unbalanced training datasets for the 50 learnwares that contained only some of the categories. This makes it unlikely that there exists any learnware in the learnware market that can accurately handle all categories of data; only the learnware whose training data is closest to the data distribution of the target task is likely to perform well on the target task. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with a non-zero probability of sampling on only 4 categories, and the sampling ratio is 0.4: 0.4: 0.1: 0.1. Ultimately, the training set for each learnware contains 12,000 samples covering the data of 4 categories in CIFAR-10.

We constructed 50 target tasks using data from the test set of CIFAR-10. Similar to constructing the training set for the learnwares, to allow for some variation between tasks, we sampled the test set unevenly. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with non-zero sampling probability on 6 categories, and the sampling ratio is 0.3: 0.3: 0.1: 0.1: 0.1: 0.1. Ultimately, each target task contains 3000 samples covering the data of 6 categories in CIFAR-10.

With this experimental setup, we evaluated the performance of RKME Image using 1 - Accuracy as the loss.

| Setting                           | Accuracy |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.655    |
| Best in Market (Single)           | 0.304    |
| Top-1 Reuse (Single)              | 0.406    |
| Job Selector Reuse (Multiple)     | 0.406    |
| Average Ensemble Reuse (Multiple) | 0.310    |

In some specific settings, the user will have a small number of labelled samples. In such settings, learning the weight of selected learnwares on a limited number of labelled samples can result in better performance than training directly on a limited number of labelled samples.

<div align=center>
  <img src="./docs/_static/img/image_labeled.svg"  width="50%"/>
</div>

## Text Data Experiment

### Datasets

We conducted experiments on the widely used text benchmark dataset: [20-newsgroup](http://qwone.com/~jason/20Newsgroups/). 20-newsgroup is a renowned text classification benchmark with a hierarchical structure, featuring 5 superclasses {comp, rec, sci, talk, misc}.

In the submitting stage, we enumerated all combinations of three superclasses from the five available, randomly sampling 50% of each combination from the training set to create datasets for 50 uploaders.

In the deploying stage, we considered all combinations of two superclasses out of the five, selecting all data for each combination from the testing set as a test dataset for one user. This resulted in 10 users. The user's own training data was generated using the same sampling procedure as the user test data, despite originating from the training dataset.

Model training comprised two parts: the first part involved training a tfidf feature extractor, and the second part used the extracted text feature vectors to train a naive Bayes classifier.

Our experiments comprise two components:

- **unlabeled_text_example** is designed to evaluate performance when users possess only testing data, searching and reusing learnware available in the market.
- **labeled_text_example** aims to assess performance when users have both testing and limited training data, searching and reusing learnware directly from the market instead of training a model from scratch. This helps determine the amount of training data saved for the user.

### Results

- **unlabeled_text_example**:

The table below presents the mean accuracy of search and reuse across all users:

| Setting                           | Accuracy |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.507    |
| Best in Market (Single)           | 0.859    |
| Top-1 Reuse (Single)              | 0.846    |
| Job Selector Reuse (Multiple)     | 0.845    |
| Average Ensemble Reuse (Multiple) | 0.862    |

- **labeled_text_example**:

We present the change curves in classification error rates for both the user's self-trained model and the multiple learnware reuse (EnsemblePrune), showcasing their performance on the user's test data as the user's training data increases. The average results across 10 users are depicted below:

<div align=center>
  <img src="./docs/_static/img/text_labeled.svg"  width="50%"/>
</div>

From the figure above, it is evident that when the user's own training data is limited, the performance of multiple learnware reuse surpasses that of the user's own model. As the user's training data grows, it is expected that the user's model will eventually outperform the learnware reuse. This underscores the value of reusing learnware to significantly conserve training data and achieve superior performance when user training data is limited.

# About

## Contributor
We appreciate all contributions and thank all the contributors!

<div align=center>
  <img src="https://github.com/Learnware-LAMDA/Learnware/graphs/contributors"/>
</div>

## About us

Visit [LAMDA's official website](http://www.lamda.nju.edu.cn/).
