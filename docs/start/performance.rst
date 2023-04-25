.. _performance:
================================
Experiments
================================

This chapter will introduce related experiments to illustrate the search and reuse performance of our learnware system.


Environment
================
For all experiments, we used a single linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

====================  ====================  ===============================
System                GPU                   CPU
====================  ====================  ===============================
Ubuntu 20.04.4 LTS    Nvidia Tesla V100S    Intel(R) Xeon(R) Gold 6240R
====================  ====================  ===============================



Experiments
================

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
PFS                                         
M5                                          9~15
CIFAR10               9000*3*32*32          7~10
====================  ====================  =================================

The accuracy of search and reuse is shown in the table below:

====================  ==================== ================================= =================================
Dataset               Top-1 Performance    Job Selector Reuse                Average Ensemble Reuse
====================  ==================== ================================= =================================
PFS
M5                      2.066 +/- 0.424    2.116 +/- 0.472                    2.512 +/- 0.573
CIFAR10                 0.619 +/- 0.138    0.585 +/- 0.056                    0.715 +/- 0.075
====================  ==================== ================================= =================================


Get Start Examples
=========================
Examples for `PFS, M5` and `CIFAR10` are available at [xxx]. You can run { main.py } directly to reproduce related experiments.
The test code is mainly composed of three parts, namely data preparation (optional), specification generation and market construction, and search test.
You can load data prepared by as and skip the data preparation step.