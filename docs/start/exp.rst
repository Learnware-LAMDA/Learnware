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

Datasets
------------------
We conducted experiments on the widely used text benchmark dataset: `20-newsgroup <http://qwone.com/~jason/20Newsgroups/>`_.
20-newsgroup is a renowned text classification benchmark with a hierarchical structure, featuring 5 superclasses {comp, rec, sci, talk, misc}.

In the submitting stage, we enumerated all combinations of three superclasses from the five available, randomly sampling 50% of each combination from the training set to create datasets for 50 uploaders.

In the deploying stage, we considered all combinations of two superclasses out of the five, selecting all data for each combination from the testing set as a test dataset for one user. This resulted in 10 users.
The user's own training data was generated using the same sampling procedure as the user test data, despite originating from the training dataset.

Model training comprised two parts: the first part involved training a tfidf feature extractor, and the second part used the extracted text feature vectors to train a naive Bayes classifier.

Our experiments comprises two components:

* ``test_unlabeled`` is designed to evaluate performance when users possess only testing data, searching and reusing learnware available in the market.
* ``test_labeled`` aims to assess performance when users have both testing and limited training data, searching and reusing learnware directly from the market instead of training a model from scratch. This helps determine the amount of training data saved for the user.

Results
----------------

* ``test_unlabeled``:

The accuracy of search and reuse is presented in the table below:

==================== ================================= =================================
 Top-1 Performance         Job Selector Reuse                Average Ensemble Reuse
==================== ================================= =================================
  0.859 +/- 0.051          0.844 +/- 0.053                    0.858 +/- 0.051
==================== ================================= =================================

* ``test_labeled``:

We present the change curves in classification error rates for both the user's self-trained model and the multiple learnware reuse(EnsemblePrune), showcasing their performance on the user's test data as the user's training data increases. The average results across 10 users are depicted below:

.. image:: ../_static/img/text_example_labeled_curves.png
   :width: 300
   :height: 200
   :alt: Text Limited Labeled Data


From the figure above, it is evident that when the user's own training data is limited, the performance of multiple learnware reuse surpasses that of the user's own model. As the user's training data grows, it is expected that the user's model will eventually outperform the learnware reuse. This underscores the value of reusing learnware to significantly conserve training data and achieve superior performance when user training data is limited.


Image Experiment
====================


Get Start Examples
=========================
Examples for `PFS, M5` and `CIFAR10` are available at [xxx]. You can run { main.py } directly to reproduce related experiments.
The test code is mainly composed of three parts, namely data preparation (optional), specification generation and market construction, and search test.
You can load data prepared by as and skip the data preparation step.


Examples for the `20-newsgroup` dataset are available at [examples/dataset_text_workflow].
We utilize the `fire` module to construct our experiments. You can execute the experiment with the following commands:

* `python main.py prepare_market`: Prepares the market.
* `python main.py test_unlabeled`: Executes the test_unlabeled experiment; the results will be printed in the terminal.
* `python main.py test_labeled`: Executes the test_labeled experiment; result curves will be automatically saved in the `figs` directory.
* Additionally, you can use `python main.py test_unlabeled True` to combine steps 1 and 2. The same approach applies to running test_labeled directly.