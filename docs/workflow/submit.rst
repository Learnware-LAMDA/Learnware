==========================================
Learnware Preparation and Submission
==========================================

In this section, we provide a comprehensive guide on submitting your custom learnware to the Learnware Market.
We will first discuss the necessary components of a valid learnware, followed by a detailed explanation on how to upload and remove learnwares within ``Learnware Market``.


Prepare Learnware
====================

A valid learnware is encapsulated in a zipfile, comprising four essential components.
Below, we illustrate the detailed structure of a learnware zipfile.

``__init__.py``
---------------

Within ``Learnware Market``, every uploader must provide a unified set of interfaces for their model, 
facilitating easy utilization for future users.
The ``__init__.py`` file serves as the Python interface for your model's fitting, prediction, and fine-tuning processes.
For example, the code snippet below is used to train and save a SVM model for a sample dataset on sklearn digits classification:

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


Then the corresponding ``__init__.py`` for this SVM model should be structured as follows:

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
    
Please remember to specify the ``input_shape`` and ``output_shape`` corresponding to your model. 
In our sklearn digits classification example, these would be (64,) and (10,) respectively.


``stat.json``
-------------

To accurately and effectively match users with appropriate learnwares for their tasks, we require information about your training dataset.
Specifically, you are required to provide a statistical specification 
stored as a json file, such as ``stat.json``, which contains the statistical information of the dataset. 
This json file meets all our requirements regarding your training data, so you don't need to upload the actual data.

There are various methods to generate a statistical specification.
If you choose to use Reduced Kernel Mean Embedding (RKME) as your statistical specification, 
the following code snippet offers guidance on how to construct and store the RKME of a dataset:

.. code-block:: python
    
    import learnware.specification as specification
    
    # generate rkme specification for digits dataset
    spec = specification.utils.generate_rkme_spec(X=data_X)
    spec.save("stat.json")


``learnware.yaml``
------------------

Additionally, you are asked to prepare a configuration file in YAML format.
The file should detail your model's class name, the type of statistical specification(e.g. Reduced Kernel Mean Embedding, ``RKMEStatSpecification``), and 
the file name of your statistical specification file. The following ``learnware.yaml`` provides an example of
how your learnware configuration file should be structured, based on our previous discussion:

.. code-block:: yaml

    model:
      class_name: SVM
      kwargs: {}
    stat_specifications:
      - module_path: learnware.specification
        class_name: RKMEStatSpecification
        file_name: stat.json
        kwargs: {}  


``environment.yaml`` or ``requirements.txt``
--------------------------------------------

In order to allow others to execute your learnware, it's necessary to specify your model's dependencies. 
You can do this by providing either an ``environment.yaml`` file or a ``requirements.txt`` file.


- ``environment.yaml`` for conda:

   If you provide an ``environment.yaml``, a new conda environment will be created based on this file 
   when users install your learnware. You can generate this yaml file using the following command:

    .. code-block::

        conda env export | grep -v "^prefix: " > environment.yaml


- ``requirements.txt`` for pip:

    If you provide a ``requirements.txt``, the dependent packages will be installed using the `-r` option of pip.
    You can find more information about ``requirements.txt`` in 
    `pip documentation <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_.
    
        
We recommend using ``environment.yaml`` as it can help minimize conflicts between different packages.

.. note::
    Whether you choose to use ``environment.yaml`` or ``requirements.txt``, 
    it's important to keep your dependencies as minimal as possible. 
    This may involve manually opening the file and removing any unnecessary packages.


Upload Learnware 
==================

After preparing the four required files mentioned above, 
you can bundle them into your own learnware zipfile. Along with the generated semantic specification that 
succinctly describes the features of your task and model (for more details, please refer to :ref:`semantic specification<components/spec:Semantic Specification>`), 
you can effortlessly upload your learnware to the ``Learnware Market`` using a single line of code:

.. code-block:: python

    import learnware
    from learnware.market import EasyMarket

    learnware.init()
    
    # EasyMarket: most basic set of functions in a Learnware Market
    easy_market = EasyMarket(market_id="demo", rebuild=True) 
    
    # single line uploading
    easy_market.add_learnware(zip_path, semantic_spec) 

Here, ``zip_path`` refers to the directory of your learnware zipfile.


Remove Learnware
==================

As administrators of the ``Learnware Market``, it's crucial to remove learnwares that exhibit suspicious uploading motives.
Once you have the necessary permissions and approvals, you can use the following code to remove a learnware 
from the ``Learnware Market``:

.. code-block:: python

    easy_market.delete_learnware(learnware_id)

Here,  ``learnware_id`` refers to the market ID of the learnware to be removed.