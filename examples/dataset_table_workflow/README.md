# Tabular Dataset Workflow Example

## Introduction

On various tabular datasets, we initially evaluate the performance of identifying and reusing learnwares from the learnware market that share the same feature space as the user's tasks. Additionally, since tabular tasks often come from heterogeneous feature spaces, we also assess the identification and reuse of learnwares from different feature spaces.

### Settings

Our study utilize three public datasets in the field of sales forecasting: [Predict Future Sales (PFS)](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data), [M5 Forecasting (M5)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), and [Corporacion](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data). To enrich the data, we apply diverse feature engineering methods to these datasets. Then we divide each dataset by store and further split the data for each store into training and test sets. A LightGBM is trained on each Corporacion and PFS training set, while the test sets and M5 datasets are reversed to construct user tasks. This results in an experimental market consisting of 265 learnwares, encompassing five types of feature spaces and two types of label spaces. All these learnwares have been uploaded to the [Beimingwu system](https://bmwu.cloud/).

### Baseline algorithms
The most basic way to reuse a learnware is Top-1 reuser, which directly uses the single learnware chosen by RKME specification. Besides, we implement two data-free reusers and two data-dependent reusers that works on single or multiple helpful learnwares identified from the market. When users have no labeled data, JobSelector reuser selects different learnwares for different samples by training a job selector classifier; AverageEnsemble reuser uses an ensemble method to make predictions. In cases where users possess both test data and limited labeled training data, EnsemblePruning reuser selectively ensembles a subset of learnwares to choose the ones that are most suitable for the user’s task; FeatureAugment reuser regards each received learnware as a feature augmentor, taking its output as a new feature and then builds a simple model on the augmented feature set. JobSelector and FeatureAugment are only effective for tabular data, while others are also useful for text and image data.

## Homogeneous Cases

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
  <img src="../../docs/_static/img/Homo_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

## Heterogeneous Cases

Based on the similarity of tasks between the market's learnwares and the users, the heterogeneous cases can be further categorized into different feature engineering and different task scenarios.

### Different Feature Engineering Scenarios

We consider the 41 stores within the PFS dataset as users, generating their user data using a unique feature engineering approach that differ from the methods employed by the learnwares in the market. As a result, while some learnwares in the market are also designed for the PFS dataset, the feature spaces do not align exactly. 

In this experimental setup, we examine various data-free reusers. The results in the following table indicate that even when users lack labeled data, the market exhibits strong performance, particularly with the AverageEnsemble method that reuses multiple learnwares.

<div align=center>

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 1.149  |
| Best in Market (Single)           | 1.038  |
| Top-1 Reuse (Single)              | 1.105  |
| Average Ensemble Reuse (Multiple) | 1.081  |

</div>


### Different Task Scenarios

We employ three distinct feature engineering methods on all the ten stores from the M5 dataset, resulting in a total of 30 users. Although the overall task of sales forecasting aligns with the tasks addressed by the learnwares in the market, there are no learnwares specifically designed to satisfy the M5 sales forecasting requirements. 

In the following figure, we present the loss curves for the user's self-trained model and several learnware reuse methods. It is evident that heterogeneous learnwares prove beneficial with a limited amount of the user's labeled data, facilitating better alignment with the user's specific task. 

<div align=center>
  <img src="../../docs/_static/img/Hetero_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

## Reproduction

Run the following command to get the table results in `Homogeneous Cases`:

```bash
python workflow.py unlabeled_homo_table_example
```

Run the following command to get the figure results in `Homogeneous Cases`:

```bash
python workflow.py labeled_homo_table_example
```

Run the following command to get the table results in `Heterogeneous Cases`:

```bash
python workflow.py cross_feat_eng_hetero_table_example
```

Run the following command to get the figure results in `Heterogeneous Cases`:

```bash
python workflow.py cross_task_hetero_table_example
```