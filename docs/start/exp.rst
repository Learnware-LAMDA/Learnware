.. _exp:
================================
Experiments and Examples
================================

This chapter will introduce related experiments to illustrate the search and reuse performance of our learnware system.

Environment
====================
For all experiments, we used a single linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

====================  ====================  ===============================
System                GPU                   CPU
====================  ====================  ===============================
Ubuntu 20.04.4 LTS    Nvidia Tesla V100S    Intel(R) Xeon(R) Gold 6240R
====================  ====================  ===============================


Table: homo+hetero
====================

Datasets
------------------
We designed experiments on three publicly available datasets, namely `Prediction Future Sales (PFS) <https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data>`_,
`M5 Forecasting (M5) <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>`_ and `CIFAR 10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
For the two sales forecasting data sets of PFS and M5, we divide the user data according to different stores, and train the Ridge model and LightGBM model on the corresponding data respectively.
For the CIFAR10 image classification task, we first randomly pick 6 to 10 categories, and randomly select 800 to 2000 samples from each category from the categories corresponding to the training set, constituting a total of 50 different uploaders.
For test users, we first randomly pick 3 to 6 categories, and randomly select 150 to 350 samples from each category from the corresponding categories from the test set, constituting a total of 20 different users.

We tested the efficiency of the specification generation and the accuracy of the search and reuse model respectively.
The evaluation index on PFS and M5 data is RMSE, and the evaluation index on CIFAR10 classification task is classification accuracy

Results
----------------

The time-consuming specification generation is shown in the table below:

====================  ====================  =================================
Dataset               Data Dimensions       Specification Generation Time (s)
====================  ====================  =================================
PFS                   8714274*31            < 1.5
M5                    46027957*82           9~15
CIFAR10               9000*3*32*32          7~10
====================  ====================  =================================

The accuracy of search and reuse is shown in the table below:

====================  ==================== ================================= =================================
Dataset               Top-1 Performance    Job Selector Reuse                Average Ensemble Reuse
====================  ==================== ================================= =================================
PFS                     1.955 +/- 2.866    2.175 +/- 2.847                    1.950 +/- 2.888
M5                      2.066 +/- 0.424    2.116 +/- 0.472                    2.512 +/- 0.573
CIFAR10                 0.619 +/- 0.138    0.585 +/- 0.056                    0.715 +/- 0.075
====================  ==================== ================================= =================================



Text Experiment
====================


Image Experiment
====================

For the CIFAR-10 dataset, we sampled the training set unevenly by category and constructed unbalanced training datasets for the 50 learnwares that contained only some of the categories. This makes it unlikely that there exists any learnware in the learnware market that can accurately handle all categories of data; only the learnware whose training data is closest to the data distribution of the target task is likely to perform well on the target task. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with a non-zero probability of sampling on only 4 categories, and the sampling ratio is 0.4: 0.4: 0.1: 0.1. Ultimately, the training set for each learnware contains 12,000 samples covering the data of 4 categories in CIFAR-10.

We constructed 50 target tasks using data from the test set of CIFAR-10. Similar to constructing the training set for the learnwares, in order to allow for some variation between tasks, we sampled the test set unevenly. Specifically, the probability of each category being sampled obeys a random multinomial distribution, with non-zero sampling probability on 6 categories, and the sampling ratio is 0.3: 0.3: 0.1: 0.1: 0.1: 0.1. Ultimately, each target task contains 3000 samples covering the data of 6 categories in CIFAR-10.

With this experimental setup, we evaluated the performance of RKME Image using 1 - Accuracy as the loss.

==================== ==================== ==================== ====================
 Top-1 Reuse         Job Selector Reuse    Voting Reuse          Best in Market    
==================== ==================== ==================== ====================
 0.406 +/- 0.128      0.406 +/- 0.128      0.310 +/- 0.112       0.304 ± 0.046     
==================== ==================== ==================== ====================

In some specific settings, the user will have a small number of labelled samples. In such settings, learning the weight of selected learnwares on a limited number of labelled samples can result in a better performance than training directly on a limited number of labelled samples.

.. image:: ../_static/img/image_labeled.png
   :align: center

Get Start Examples
=========================
Examples for `PFS, M5` and `CIFAR10` are available at [xxx]. You can run { main.py } directly to reproduce related experiments.
The test code is mainly composed of three parts, namely data preparation (optional), specification generation and market construction, and search test.
You can load data prepared by as and skip the data preparation step.