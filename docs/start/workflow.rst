================
Workflow
================


Submit learnware
=================

In this section, a detailed guide on how to submit your own learnware to the Learnware Market will be provided.
Specifically, we will first elaborate on the components that a valid learnware file should include, then explain
how learnwares are uploaded and removed within ``Learnware Market``.


Prepare Learnware
-------------------

A valid learnware is a zipfile which consists of four essentia parts. Here we demonstrate the detail format of a learnware zipfile.

``__init__.py``
^^^^^^^^^^^^^^^

In ``Learnware Market``, each uploader is required to provide a set of unified interfaces for their model, 
which enables convenient usage by future users.
``__init__.py`` is the python file offering interfaces for your model's fitting, predicting and fine-tuning. For example,
the code snippet below trains and saves a SVM model for a sample dataset on sklearn digits classification:

.. code-block:: python

    import joblib
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X,y = load_digits(return_X_y=True) 
    data_X, _, data_y, _ = train_test_split(X, y, test_size=0.3, shuffle=True)

    # input dimension: (64, ), output dimension: (10, )
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(data_X, data_y)

    joblib.dump(clf, "svm.pkl") # model is stored as file "svm.pkl"


Then the ``__init__.py`` for this SVM model should be as follows:

.. code-block:: python
    
    import os
    import joblib
    import numpy as np
    from learnware.model import BaseModel


    class SVM(BaseModel):
        def __init__(self):
            super(SVM, self).__init__(input_shape=(64,), output_shape=(10,))
            dir_path = os.path.dirname(os.path.abspath(__file__))
            self.model = joblib.load(os.path.join("svm.pkl"))

        def fit(self, X: np.ndarray, y: np.ndarray):
            pass

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(X)

        def finetune(self, X: np.ndarray, y: np.ndarray):
            pass
    
As a kind reminder, don't forget to fill in ``input_shape`` and ``output_shape`` corresponding to the model 
(in our sklearn digits classification example, (64,) and (10,) respectively).


``stat.json``
^^^^^^^^^^^^^^^

In order to better match users with learnwares suitable for their tasks, 
we do need the information of your training dataset. Specifically, you need to provide a statistical specification 
stored as a json file, e.g., ``stat.json``, which contains statistical information of the dataset. 
This json file is all we required regarding your training data, and there is no need for you to upload your own data.

Statistical specification can have many implementation approaches. 
If Reduced Kernel Mean Embedding (RKME) is chosen to be as statistical specification, 
the following code snippet provides guidance on how to build and store the RKME of a dataset:

.. code-block:: python
    
    import learnware.specification as specification
    
    # generate rkme specification for digits dataset
    spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
    spec.save(os.path.join("stat.json"))


``learnware.yaml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also require you to prepare a configuration file in YAML format,
which describes your model class name, type of statistical specification(e.g. Reduced Kernel Mean Embedding, ``RKMEStatSpecification``), and 
the file name of your statistical specification file. The following ``learnware.yaml`` demonstrates 
how your learnware configuration file should be structured based on our previous example:

.. code-block:: yaml

    model:
      class_name: SVM
      kwargs: {}
    stat_specifications:
      - module_path: learnware.specification
        class_name: RKMEStatSpecification
        file_name: stat.json
        kwargs: {}  


``environment.yaml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this YAML file, you need to specify the conda environment configuration for running your model 
(if the model environment is incompatible, you can rely on this for manual configuration). 
You can generate this file according to the following steps:

- Create env config for conda:

    .. code-block::

        conda env export | grep -v "^prefix: " > environment.yaml
        
- Recover env from config:

    .. code-block::

        conda env create -f environment.yaml


Upload Learnware 
-------------------

Once you have prepared the four required files mentioned above, 
you can package them as your own learnware zipfile. Combined with the generated semantic specification that 
briefly describes the features of your task and model (Please refer to :ref:`semantic_specification` for more details), 
you can easily upload your learnware to the ``Learnware Market`` with a single line of code:

.. code-block:: python

    import learnware
    from learnware.market import EasyMarket

    learnware.init()
    
    # EasyMarket: most basic set of functions in a Learnware Market
    easy_market = EasyMarket(market_id="demo", rebuild=True) 
    
    # single line uploading
    easy_market.add_learnware(zip_path, semantic_spec) 

Here, ``zip_path`` is the directory of your learnware zipfile.


Remove Learnware
------------------

As ``Learnware Market`` administrators, it is necessary to remove learnwares with suspicious uploading motives.
With required permissions and approvals, you can use the following code to remove a learnware 
from the ``Learnware Market``:

.. code-block:: python

    easy_market.delete_learnware(learnware_id)

Here,  ``learnware_id`` is the market ID of the learnware to be removed.



Identify helpful learnware
===========================

When a user comes with her requirements, the market should identify helpful learnwares and recommend them to the user.
The search of helpful learnwares is based on the user information, and can be divided into two stages: semantic specification search and statistical specification search.

User information
-------------------------------
The user should provide her requirements in ``BaseUserInfo``. The class ``BaseUserInfo`` consists of user's semantic specification ``user_semantic`` and statistical information ``stat_info``. 

The semantic specification ``user_semantic`` is stored in a ``dict``, with keywords 'Data', 'Task', 'Library', 'Scenario', 'Description' and 'Name'. An example is shown below, and you could choose their values according to the figure. For the keys of type 'Class, you should choose one ilegal value; for the keys of type 'Tag', you can choose one or more values; for the keys of type 'String', you should provide a string; the key 'Description' is used in learnwares' semantic specifications and is ignored in user semantic specification; the values of all these keys can be empty if the user have no idea of them.

.. code-block:: python

    # An example of user_semantic
    user_semantic = {
        "Data": {"Values": ["Image"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Tag"},
        "Scenario": {"Values": ["Education"], "Type": "Class"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "digits", "Type": "String"},
    }

.. _semantic_specification:

.. figure: ..\_static\img\semantic_spec.png
   :alt: Semantic Specification
   :align: center

引用方式 :ref:`semantic_specification` 。


The user's statistical information ``stat_info`` is stored in a ``json`` file, e.g., ``stat.json``. The generation of this file is seen in `这是一个语义规约生成的链接`_.



Semantic Specification Search
-------------------------------
To search for learnwares that fit your task purpose, 
the user should first provide a semantic specification ``user_semantic`` that describes the characteristics of your task.
The Learnware Market will perform a first-stage search based on ``user_semantic``,
identifying potentially helpful leranwares whose models solve tasks similar to your requirements. 

.. code-block:: python

    # construct user_info which includes semantic specification for searching learnware
    user_info = BaseUserInfo(semantic_spec=semantic_spec)

    # search_learnware performs semantic specification search if user_info doesn't include a statistical specification
    _, single_learnware_list, _ = easy_market.search_learnware(user_info) 

    # single_learnware_list is the learnware list by semantic specification searching
    print(single_learnware_list)

In semantic specification search, we go through all learnwares in the market to compare their semantic specifications with the user's one, and return all the learnwares that pass through the comparation. When comparing two learnwares' semantic specifications, we design different ways for different semantic keys:

- For semantic keys with type 'Class', they are matched only if they have the same value.

- For semantic keys with type 'Tag', they are matched only if they have nonempty intersections.

- For the user's input in the search box, it matchs with a learnware's semantic specification only if it's a substring of its 'Name' or 'Description'. All the strings are converted to the lower case before matching.

- When a key value is missing, it will not participate in the match. The user could upload no semantic specifications if he wants.

Statistical Specification Search
---------------------------------

If you choose to provide your own statistical specification file ``stat.json``, 
the Learnware Market can perform a more accurate leanware selection from 
the learnwares returned by the previous step. This second-stage search is based on statistical information and returns one or more learnwares that are most likely to be helpful for your task. 

For example, the following code is designed to work with Reduced Kernel Mean Embedding (RKME) as a statistical specification:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.rkme.RKMEStatSpecification()
    user_spec.load(os.path.join("rkme.json"))
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

The return values of statistical specification search are ``sorted_score_list``, ``single_learnware_list``, ``mixture_score`` and ``mixture_learnware_list``.
``sorted_score_list`` and ``single_learnware_list`` are the ranking of each single learnware and the corresponding scores. We return at least 15 learnwares unless there're no enough ones. If there are more than 15 matched learnwares, the ones with scores less than 50 will be ignored.
``mixture_score`` and ``mixture_learnware_list`` are the chosen mixture learnwares and the corresponding score. At most 5 learnwares will be chosen, whose mixture may have a relatively good performance on the user's task.


The statistical specification search is done in the following way.
We first filter by the dimension of RKME specifications; only those with the same dimension with the user's will enter the subsequent stage.

The single_learnware_list is calculated using the distances between two RKMEs. The greater the distance from the user's RKME, the lower the score is. 

The mixture_learnware_list is calculated in a greedy way. Each time we choose a learnware to make their mixture closer to the user's RKME. Specifically, each time we go through all the left learnwares to find the one whose combination with chosen learnwares could minimize the distance between their mixture's RKME and the user's RKME. The mixture weight is calculated by minimizing the RKME distance, which is solved by quadratic programming. If the distance become larger or the number of chosen learnwares reaches a threshold, the process will end and the chosen learnware and weight list will return.



Reuse learnware
===========================

This part introduces two baseline methods for reusing a given list of learnwares, namely ``JobSelectorReuser`` and ``AveragingReuser``.
Instead of training a model from scratch, the user can easily reuse a list of learnwares (``List[Learnware]``) to predict the labels of their own data (``numpy.ndarray`` or ``torch.Tensor``).

To illustrate, we provide a code demonstration that obtains the user dataset using ``sklearn.datasets.load_digits``, where ``test_data`` represents the data that requires prediction.
Assuming that ``learnware_list`` is the list of learnwares searched by the learnware market based on user specifications, the user can reuse each learnware in the ``learnware_list`` through ``JobSelectorReuser`` or ``AveragingReuser`` to predict the label of ``test_data``, thereby avoiding training a model from scratch.

.. code-block:: python

    from sklearn.datasets import load_digits
    from learnware.learnware import JobSelectorReuser, AveragingReuser

    # Load user data
    X, y = load_digits(return_X_y=True)
    test_data = X

    # Based on user information, the learnware market returns a list of learnwares (learnware_list)
    # Use jobselector reuser to reuse the searched learnwares to make prediction
    reuse_job_selector = JobSelectorReuser(learnware_list=learnware_list)
    job_selector_predict_y = reuse_job_selector.predict(user_data=test_data)

    # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
    reuse_ensemble = AveragingReuser(learnware_list=learnware_list)
    ensemble_predict_y = reuse_ensemble.predict(user_data=test_data)


JobSelectorReuser
-------------------

The ``JobSelectorReuser`` is a class that inherits from the base reuse class ``BaseReuser``.
Its purpose is to create a job selector that identifies the optimal learnware for each data point in user data.
There are three parameters required to initialize the class:

- ``learnware_list``: A list of objects of type ``Learnware``. Each ``Learnware`` object should have an RKME specification.
- ``herding_num``: An optional integer that specifies the number of items to herd, which defaults to 1000 if not provided.
- ``use_herding``: A boolean flag indicating whether to use kernel herding.

The job selector is essentially a multi-class classifier :math:`g(\boldsymbol{x}):\mathcal{X}\rightarrow \mathcal{I}` with :math:`\mathcal{I}=\{1,\ldots, C\}`, where :math:`C` is the size of ``learnware_list``.
Given a testing sample :math:`\boldsymbol{x}`, the ``JobSelectorReuser`` predicts it by using the :math:`g(\boldsymbol{x})`-th learnware in ``learnware_list``.
If ``use_herding`` is set to false, the ``JobSelectorReuser`` uses data points in each learware's RKME spefication with the corresponding learnware index to train a job selector.
If ``use_herding`` is true, the algorithm estimates the mixture weight based on RKME specifications and raw user data, uses the weight to generate ``herding_num`` auxiliary data points mimicking the user distribution through the kernel herding method, and learns a job selector on these data.


AveragingReuser
-------------------

The ``AveragingReuser`` is a class that inherits from the base reuse class ``BaseReuser``, that implements the average ensemble method by averaging each learnware's output to predict user data.
There are two parameters required to initialize the class:

- ``learnware_list``: A list of objects of type ``Learnware``.
- ``mode``: The mode of averaging leanrware outputs, which can be set to "mean" or "vote" and defaults to "mean".

If ``mode`` is set to "mean", the ``AveragingReuser`` computes the mean of the learnware's output to predict user data, which is commonly used in regression tasks.
If ``mode`` is set to "vote", the ``AveragingReuser`` computes the mean of the softmax of the learnware's output to predict each label probability of user data, which is commonly used in classification tasks.
