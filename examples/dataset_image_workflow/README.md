# Image Dataset Workflow Example

## Introduction

We conducted experiments on the widely used image benchmark dataset: [``CIFAR-10``](https://www.cs.toronto.edu/~kriz/cifar.html).
The  ``CIFAR-10`` dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.  

In the submitting stage, we sampled the training set non-uniformly by category, and constructed unbalanced training datasets for the 50 learnwares that contained only part of the categories randomly. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with positive sampling probability on only 4 categories, and a sampling ratio of 0.4: 0.4: 0.1: 0.1. The training set for each learnware contains 12,500 samples covering data from the 4 categories in CIFAR-10.

In the deploying stage, we constructed 100 user tasks using the CIFAR-10 test set data. Similar to constructing the training set, the probability of each category being sampled obeys a random multinomial distribution, with positive sampling probabilities on only 6 categories, with a sampling ratio of 0.3: 0.3: 0.1: 0.1: 0.1: 0.1. Each user task contains 3,000 samples covering the data of 6 categories in CIFAR-10. 

Our example ``image_example`` shows the performance in two different scenarios:

**Unlabelled Sample Scenario**: This scenario is designed to evaluate performance when users possess only testing data, searching and reusing learnware available in the market.

**Labelled Sample Scenario**: This scenario aims to assess performance when users have both testing and limited training data, searching and reusing learnware directly from the market instead of training a model from scratch. This helps determine the amount of training data saved for the user.

## Run the code

Run the following command to start the ``image_example``.

```bash
python workflow.py image_example
```

## Results

With the experimental setup above, we evaluated the performance of RKME Image by calculating the mean accuracy across all users.

### Unlabelled Sample Scenario

| Metric                               | Value               |
|--------------------------------------|---------------------|
| Mean in Market (Single)              | 0.346               |
| Best in Market (Single)              | 0.688               |
| Top-1 Reuse (Single)                 | 0.534               |
| Job Selector Reuse (Multiple)        | 0.534               |
| Average Ensemble Reuse (Multiple)    | 0.676               |

### Labelled Sample Scenario

In some specific settings, the user will have a small number of labeled samples. In such settings, learning the weight of selected learnwares on a limited number of labeled samples can result in a better performance than training directly on a limited number of labeled samples.

<div align=center>
  <img src="../../docs/_static/img/image_labeled.svg" alt="Results on Image Experimental Scenario" style="width:50%;" />
</div>

Note that in labelled sample scenario, the labelled samples are repeatedly sampled 3 to 10 times, in order to reduce the estimation error in accuracy due to random sampling.