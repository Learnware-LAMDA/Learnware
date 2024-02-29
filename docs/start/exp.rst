.. _exp:

================================
Experiments and Examples
================================

In this section, we build various types of experimental scenarios and conduct extensive empirical study to evaluate the baseline algorithms, implemented and refined in the ``learnware`` package, for specification generation, learnware identification, and reuse on tabular, image, and text data.

Environment
====================
For all experiments, we used a single linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

====================  ====================  ===============================
System                GPU                   CPU
====================  ====================  ===============================
Ubuntu 20.04.4 LTS    Nvidia Tesla V100S    Intel(R) Xeon(R) Gold 6240R
====================  ====================  ===============================


Tabular Data Experiments
===========================

On various tabular datasets, we initially evaluate the performance of identifying and reusing learnwares from the learnware market that share the same feature space as the user's tasks. Additionally, since tabular tasks often come from heterogeneous feature spaces, we also assess the identification and reuse of learnwares from different feature spaces. 

Settings
------------------
Our study utilize three public datasets in the field of sales forecasting: `Predict Future Sales (PFS) <https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data>`_, `M5 Forecasting (M5) <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>`_ and `Corporacion <https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data>`_. To enrich the data, we apply diverse feature engineering methods to these datasets. Then we divide each dataset by store and further split the data for each store into training and test sets. A LightGBM is trained on each Corporacion and PFS training set, while the test sets and M5 datasets are reversed to construct user tasks. This results in an experimental market consisting of 265 learnwares, encompassing five types of feature spaces and two types of label spaces. All these learnwares have been uploaded to the learnware dock system.

Baseline algorithms
--------------------

The most basic way to reuse a learnware is Top-1 reuser, which directly uses the single learnware chosen by RKME specification. Besides, we implement two data-free reusers and two data-dependent reusers that works on single or multiple helpful learnwares identified from the market. When users have no labeled data, JobSelector reuser selects different learnwares for different samples by training a job selector classifier; AverageEnsemble reuser uses an ensemble method to make predictions. In cases where users possess both test data and limited labeled training data, EnsemblePruning reuser selectively ensembles a subset of learnwares to choose the ones that are most suitable for the user's task; FeatureAugment reuser regards each received learnware as a feature augmentor, taking its output as a new feature and then builds a simple model on the augmented feature set. JobSelector and FeatureAugment are only effective for tabular data, while others are also useful for text and image data.

Homogeneous Cases
------------------

In the homogeneous cases, the 53 stores within the PFS dataset function as 53 individual users. Each store utilizes its own test data as user data and applies the same feature engineering approach used in the learnware market. These users could subsequently search for homogeneous learnwares within the market that possessed the same feature spaces as their tasks.

We conduct a comparison among different baseline algorithms when the users have no labeled data or limited amounts of labeled data. The average losses over all users are illustrated in the table below. It shows that unlabeled methods are much better than random choosing and deploying one learnware from the market.


+-----------------------------------+---------------------+
| Setting                           |        MSE          |
+===================================+=====================+
| Mean in Market (Single)           |   0.897             |
+-----------------------------------+---------------------+
| Best in Market (Single)           |   0.756             |
+-----------------------------------+---------------------+
| Top-1 Reuse (Single)              |   0.830             |
+-----------------------------------+---------------------+
| Job Selector Reuse (Multiple)     |   0.848             |
+-----------------------------------+---------------------+
| Average Ensemble Reuse (Multiple) |   0.816             |
+-----------------------------------+---------------------+


The figure below showcases the results for different amounts of labeled data provided by the user; for each user, we conducted multiple experiments repeatedly and calculated the mean and standard deviation of the losses; the average losses over all users are illustrated in the figure. It illustrates that when users have limited training data, identifying and reusing single or multiple learnwares yields superior performance compared to user's self-trained models. 

.. image:: ../_static/img/Homo_labeled_curves.svg
   :align: center

Heterogeneous Tabular Dataset
------------------------------

Based on the similarity of tasks between the market's learnwares and the users, the heterogeneous cases can be further categorized into different feature engineering and different task scenarios.

Different Feature Engineering Scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We consider the 41 stores within the PFS dataset as users, generating their user data using a unique feature engineering approach that differ from the methods employed by the learnwares in the market. As a result, while some learnwares in the market are also designed for the PFS dataset, the feature spaces do not align exactly. 

In this experimental setup, we examine various data-free reusers. The results in the following table indicate that even when users lack labeled data, the market exhibits strong performance, particularly with the AverageEnsemble method that reuses multiple learnwares.


+-----------------------------------+---------------------+
| Setting                           |        MSE          |
+===================================+=====================+
| Mean in Market (Single)           | 1.459               |
+-----------------------------------+---------------------+
| Best in Market (Single)           | 1.038               |
+-----------------------------------+---------------------+
| Top-1 Reuse (Single)              | 1.075               |
+-----------------------------------+---------------------+
| Average Ensemble Reuse (Multiple) | 1.064               |
+-----------------------------------+---------------------+


Different Task Scenarios
^^^^^^^^^^^^^^^^^^^^^^^

We employ three distinct feature engineering methods on all the ten stores from the M5 dataset, resulting in a total of 30 users. Although the overall task of sales forecasting aligns with the tasks addressed by the learnwares in the market, there are no learnwares specifically designed to satisfy the M5 sales forecasting requirements. 

In the following figure, we present the loss curves for the user's self-trained model and several learnware reuse methods. It is evident that heterogeneous learnwares prove beneficial with a limited amount of the user's labeled data, facilitating better alignment with the user's specific task. 

.. image:: ../_static/img/Hetero_labeled_curves.svg
   :align: center


Image Data Experiment
=========================

Second, we assess our system on image datasets. It is worth noting that images of different sizes could be standardized through resizing, eliminating the need to consider heterogeneous feature cases.

Settings
----------------

We choose the famous image classification dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60000 32x32 color images in 10 classes. A total of 50 learnwares are uploaded: each learnware contains a convolutional neural network trained on an unbalanced subset that includs 12000 samples from four categories with a sampling ratio of $0.4:0.4:0.1:0.1$. 
A total of 100 user tasks are tested and each user task consists of 3000 samples of CIFAR-10 with six categories with a sampling ratio of $0.3:0.3:0.1:0.1:0.1:0.1$.


Results
-------------------
We assess the average performance of various methods using 1 - Accuracy as the loss metric. The following table and figure show that when users face a scarcity of labeled data or possess only a limited amount of it (less than 2000 instances), leveraging the learnware market can yield good performances.


+-----------------------------------+---------------------+
| Setting                           |        Accuracy     |
+===================================+=====================+
| Mean in Market (Single)           | 0.655               |
+-----------------------------------+---------------------+
| Best in Market (Single)           | 0.304               |
+-----------------------------------+---------------------+
| Top-1 Reuse (Single)              | 0.406               |
+-----------------------------------+---------------------+
| Job Selector Reuse (Multiple)     | 0.406               |
+-----------------------------------+---------------------+
| Average Ensemble Reuse (Multiple) | 0.310               |
+-----------------------------------+---------------------+


.. image:: ../_static/img/image_labeled_curves.svg
   :align: center

Text Data Experiment
==========================

Finally, we evaluate our system on text datasets. Text data naturally exhibit feature heterogeneity, but this issue can be addressed by applying a sentence embedding extractor.

Settings
------------------

We conduct experiments on the well-known text classification dataset: [20-newsgroup](http://qwone.com/~jason/20Newsgroups/), which consists approximately 20000 newsgroup documents partitioned across 20 different newsgroups.
Similar to the image experiments, a total of 50 learnwares are uploaded. Each learnware is trained on a subset that includes only half of the samples from three superclasses and the model in it is a tf-idf feature extractor combined with a naive Bayes classifier. We define 10 user tasks, and each of them encompasses two superclasses.

Results
----------------

The results are depicted in the following table and figure. Similarly, even when no labeled data is provided, the performance achieved through learnware identification and reuse can match that of the best learnware in the market. Additionally, utilizing the learnware market allows for a reduction of approximately 2000 samples compared to training models from scratch.

+-----------------------------------+---------------------+
| Setting                           |        Accuracy     |
+===================================+=====================+
| Mean in Market (Single)           | 0.507               |
+-----------------------------------+---------------------+
| Best in Market (Single)           | 0.859               |
+-----------------------------------+---------------------+
| Top-1 Reuse (Single)              | 0.846               |
+-----------------------------------+---------------------+
| Job Selector Reuse (Multiple)     | 0.845               |
+-----------------------------------+---------------------+
| Average Ensemble Reuse (Multiple) | 0.862               |
+-----------------------------------+---------------------+


.. image:: ../_static/img/text_labeled_curves.svg
   :align: center


Get Start Examples
=========================
Examples for `Tabular, Text` and `Image` data sets are available at `Learnware Examples <https://github.com/Learnware-LAMDA/Learnware/tree/main/examples>`_. You can run { workflow.py } directly to reproduce related experiments.
We utilize the `fire` module to construct our experiments.

Table Examples
------------------
* `python workflow.py unlabeled_homo_table_example`: Executes the unlabeled_homo_table_example experiment; the results will be printed in the terminal.
* `python workflow.py labeled_homo_table_example`: Executes the labeled_homo_table_example experiment; result curves will be automatically saved in the `figs` directory.
* `python workflow.py cross_feat_eng_hetero_table_example`: Executes the cross_feat_eng_hetero_table_example experiment; the results will be printed in the terminal.
* `python workflow.py cross_task_hetero_table_example`: Executes the cross_task_hetero_table_example experiment; result curves will be automatically saved in the `figs` directory.

Text Examples
------------------
You can execute the experiment with the following commands:

* `python workflow.py unlabeled_text_example`: Executes the unlabeled_text_example experiment; the results will be printed in the terminal.
* `python workflow.py labeled_text_example`: Executes the labeled_text_example experiment; result curves will be automatically saved in the `figs` directory.

Image Examples
------------------
You can execute the experiment with the following commands:

.. code-block:: bash
   
   python workflow.py image_example