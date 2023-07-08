==========================================
Learnware Preparation and Submission
==========================================

In this section, a detailed guide on how to submit your own learnware to the Learnware Market will be provided.
Specifically, we will first elaborate on the components that a valid learnware file should include, then explain
how learnwares are uploaded and removed within ``Learnware Market``.


Prepare Learnware
====================

A valid learnware is a zipfile which consists of four essential parts. Here we demonstrate the detail format of a learnware zipfile.

``__init__.py``
---------------

In ``Learnware Market``, each uploader is required to provide a set of unified interfaces for their model, 
which enables convenient usage by future users.
``__init__.py`` is the python file offering interfaces for your model's fitting, predicting and fine-tuning. For example,
the code snippet below trains and saves a SVM model for a sample dataset on sklearn digits classification:

.. code-block:: python

    import joblib
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True) 
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
-------------

In order to better match users with learnwares suitable for their tasks, 
we need the information of your training dataset. Specifically, you need to provide a statistical specification 
stored as a json file, e.g., ``stat.json``, which contains statistical information of the dataset. 
This json file is all we required regarding your training data, and there is no need for you to upload your own data.

There are multiple approaches to generate statistical specification.
If Reduced Kernel Mean Embedding (RKME) is chosen to be as statistical specification, 
the following code snippet provides guidance on how to build and store the RKME of a dataset:

.. code-block:: python
    
    import learnware.specification as specification
    
    # generate rkme specification for digits dataset
    spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
    spec.save(os.path.join("stat.json"))


``learnware.yaml``
------------------

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
--------------------

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
==================

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
==================

As ``Learnware Market`` administrators, it is necessary to remove learnwares with suspicious uploading motives.
With required permissions and approvals, you can use the following code to remove a learnware 
from the ``Learnware Market``:

.. code-block:: python

    easy_market.delete_learnware(learnware_id)

Here,  ``learnware_id`` is the market ID of the learnware to be removed.