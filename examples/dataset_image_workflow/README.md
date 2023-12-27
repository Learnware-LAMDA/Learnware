# Image Dataset Workflow Example

## Introduction

For the CIFAR-10 dataset, we sampled the training set unevenly by category and constructed unbalanced training datasets for the 50 learnwares that contained only some of the categories. This makes it unlikely that there exists any learnware in the learnware market that can accurately handle all categories of data; only the learnware whose training data is closest to the data distribution of the target task is likely to perform well on the target task. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with a non-zero probability of sampling on only 4 categories, and the sampling ratio is 0.4: 0.4: 0.1: 0.1. Ultimately, the training set for each learnware contains 12,000 samples covering the data of 4 categories in CIFAR-10.

We constructed 50 target tasks using data from the test set of CIFAR-10. Similar to constructing the training set for the learnwares, in order to allow for some variation between tasks, we sampled the test set unevenly. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with non-zero sampling probability on 6 categories, and the sampling ratio is 0.3: 0.3: 0.1: 0.1: 0.1: 0.1. Ultimately, each target task contains 3000 samples covering the data of 6 categories in CIFAR-10.

With this experimental setup, we evaluated the performance of RKME Image by calculating the mean accuracy across all users.

| Metric                               | Value               |
|--------------------------------------|---------------------|
| Mean in Market (Single)              | 0.346               |
| Best in Market (Single)              | 0.688               |
| Top-1 Reuse (Single)                 | 0.534               |
| Job Selector Reuse (Multiple)        | 0.534               |
| Average Ensemble Reuse (Multiple)    | 0.676               |

In some specific settings, the user will have a small number of labeled samples. In such settings, learning the weight of selected learnwares on a limited number of labeled samples can result in a better performance than training directly on a limited number of labeled samples.

<div align=center>
  <img src="../../docs/_static/img/image_labeled.png" alt="Image Limited Labeled Data" style="width:50%;" />
</div>

## Run the code

Run the following command to start the ``image_example``.

```bash
python workflow.py image_example
```