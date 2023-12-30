.. _submit:
==========================================
Learnware Preparation and Uoloading
==========================================

In this section, we provide a comprehensive guide on submitting your custom learnware to the ``Learnware Market``.
We will first discuss the necessary components of a valid learnware, followed by a detailed explanation on how to upload and remove learnwares within ``Learnware Market``.


Prepare Learnware
====================================

In learnware ``Learnware`` package, each learnware is encapsulated in a ``zip`` package, which should contain at least the following four files:

- ``learnware.yaml``: learnware configuration file.
- ``__init__.py``: methods for using the model.
- ``stat.json``: the statistical specification of the learnware. Its filename can be customized and recorded in learnware.yaml.
- ``environment.yaml`` or ``requirements.txt``: specifies the environment for the model.

To facilitate the construction of a learnware, we provide a `Learnware Template <https://www.bmwu.cloud/static/learnware-template.zip>`_ that you can use as a basis for building your own learnware.

Next, we will provide detailed explanations for the content of these four files.

Model Invocation File ``__init__.py``
-------------------------------------

To ensure that the uploaded learnware can be used by subsequent users, you need to provide interfaces for model fitting ``fit(X, y)``, prediction ``predict(X)``, and fine-tuning ``finetune(X, y)`` in ``__init__.py``. Among these interfaces, only the ```predict(X)``` interface is mandatory, while the others depend on the functionality of your model. 

Below is a reference template for the ```__init__.py``` file. Please make sure that the input parameter format (the number of parameters and parameter names) for each interface in your model invocation file matches the template below.

.. code-block:: python

    import os
    import pickle
    import numpy as np
    from learnware.model import BaseModel

    class MyModel(BaseModel):
        def __init__(self):
            super(MyModel, self).__init__(input_shape=(37,), output_shape=(1,))
            dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, "model.pkl")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        def fit(self, X: np.ndarray, y: np.ndarray):
            self.model = self.model.fit(X)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.model.predict(X)

        def finetune(self, X: np.ndarray, y: np.ndarray):
            pass


Please ensure that the ``MyModel`` class inherits from ``BaseModel`` in the ``learnware.model`` module, and specify the class name (e.g., ``MyModel``) in the ``learnware.yaml`` file later. 

Input and Output Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``input_shape`` and ``output_shape`` represent the input and output dimensions of the model, respectively. You can refer to the following guidelines when filling them out:
  - ``input_shape`` specifies a single input sample's dimension, and ``output_shape`` refers to the model's output dimension for a single sample.
  - When the data type being processed is text data, there are no specific requirements for the value of ``input_shape``, and it can be filled in as ``None``.
  - When the ``output_shape`` corresponds to tasks with variable outputs (such as object detection, text segmentation, etc.), there are no specific requirements for the value of ``output_shape``, and it can be filled in as ``None``.
  - For classification tasks, ``output_shape`` should be (1, ) if the model directly outputs predicted labels, and the sample labels need to start from 0. If the model outputs logits, ``output_shape`` should be specified as the number of classes, i.e., (class_num, ).

File Path
^^^^^^^^^^^^^^^^^^
If you need to load certain files within the zip package in the ``__init__.py`` file (and any other Python files that may be involved), please follow the method shown in the template above about obtaining the ``model_path``:
  - First, obtain the root directory path of the entire package by getting ``dir_path``.
  - - Then, based on the specific file's relative location within the package, obtain the specific file's path, ``model_path``.

Module Imports
^^^^^^^^^^^^^^^^^^
Please note that module imports between Python files within the zip package should be done using **relative imports**. For instance:

.. code-block:: python

    from .package_name import *
    from .package_name import module_name


Learnware Statistical Specification ``stat.json``
---------------------------------------------------

A learnware consists of a model and a specification. Therefore, after preparing the model, you need to generate a statistical specification for it. Specifically, using the previously installed ``Learnware`` package, you can use the training data ``train_x`` (supported types include numpy.ndarray, pandas.DataFrame, and torch.Tensor) as input to generate the statistical specification of the model.

Here is an example of the code:

.. code-block:: python

    from learnware.specification import generate_stat_spec

    data_type = "table" # Data types: ["table", "image", "text"]
    spec = generate_stat_spec(type=data_type, X=train_x)
    spec.save("stat.json")

It's worth noting that the above code only runs on your local computer and does not interact with any cloud servers or leak any local private data.

Additionally, if the model's training data is too large, causing the above code to fail, you can consider sampling the training data to ensure it's of a suitable size before proceeding with reduction generation.


Learnware Configuration File ``learnware.yaml``
-------------------------------------------------

This file is used to specify the class name (``MyModel``) in the model invocation file ``__init__.py``, the module called for generating the statistical specification (``learnware.specification``), the category of the statistical specification (``RKMETableSpecification``), and the specific filename (``stat.json``):

.. code-block:: yaml

    model:
    class_name: MyModel
    kwargs: {}
    stat_specifications:
    - module_path: learnware.specification
        class_name: RKMETableSpecification
        file_name: stat.json
        kwargs: {}

Please note that the statistical specification class name for different data types ``['table', 'image', 'text']`` is ``[RKMETableSpecification, RKMEImageSpecification, RKMETextSpecification]``, respectively.

Model Runtime Dependent File
--------------------------------------------

To ensure that your uploaded learnware can be used by other users, the ``zip`` package of the uploaded learnware should specify the model's runtime dependencies. The Beimingwu System supports the following two ways to specify runtime dependencies:
  - Provide an ``environment.yaml`` file supported by ``conda``.
  - Provide a ``requirements.txt`` file supported by ``pip``.

You can choose either method, but please try to remove unnecessary dependencies to keep the dependency list as minimal as possible.

Using ``environment.yaml`` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can export the `environment.yaml` file directly from the `conda` virtual environment using the following command:

- For Linux and macOS systems

.. code-block:: bash
    
    conda env export | grep -v "^prefix: " > environment.yaml

- For Windows systems:

.. code-block:: bash
    
    conda env export | findstr /v "^prefix: " > environment.yaml

Note that the ``environment.yaml`` file in the ``zip`` package needs to be encoded in ``UTF-8`` format. Please check the encoding format of the ``environment.yaml`` file after using the above command. Due to the ``conda`` version and system differences, you may not get a ``UTF-8`` encoded file (e.g. get a ``UTF-16LE`` encoded file). You'll need to manually convert the file to ``UTF-8``, which is supported by most text editors. The following ``Python`` code for encoding conversion is also for reference:

.. code-block:: python

    import codecs

    # Read the output file from the 'conda env export' command
    # Assuming the file name is environment.yaml and the export format is UTF-16LE
    with codecs.open('environment.yaml', 'r', encoding='utf-16le') as file:
        content = file.read()

    # Convert the content to UTF-8 encoding
    output_content = content.encode('utf-8')

    # Write to UTF-8 encoded file
    with open('environment.yaml', 'wb') as file:
        file.write(output_content)


Additionally, due to the complexity of users' local ``conda`` virtual environments, you can execute the following command before uploading to confirm that there are no dependency conflicts in the ``environment.yaml`` file:

.. code-block:: bash
    
    conda env create --name test_env --file environment.yaml

The above command will create a virtual environment based on the ``environment.yaml`` file, and if successful, it indicates that there are no dependency conflicts. You can delete the created virtual environment using the following command:

.. code-block:: bash

    conda env remove --name test_env

Using `requirements.txt` File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``requirements.txt`` file should list the packages required for running the ``__init__.py`` file and their specific versions. You can obtain these version details by executing the ``pip show <package_name>`` or ``conda list <package_name>`` command. Here is an example file:

.. code-block:: text

    numpy==1.23.5
    scikit-learn==1.2.2

Manually listing these dependencies can be cumbersome, so you can also use the ``pipreqs`` package to automatically scan your entire project and export the packages used along with their specific versions (though some manual verification may be required):

.. code-block:: bash

    pip install pipreqs
    pipreqs ./  # Run this command in the project's root directory

Please note that if you use the ``requirements.txt`` file to specify runtime dependencies, the system will by default install these dependencies in a ``conda`` virtual environment running ``Python 3.8`` during the learnware deployment.

Furthermore, for version-sensitive packages like ``torch``, it's essential to specify package versions in the ``requirements.txt`` file to ensure successful deployment of the uploaded learnware on other machines.

Upload Learnware
==================================

After preparing the four required files mentioned above, you can bundle them into your own learnware ``zip`` package.

Prepare Sematic Specifcation
-----------------------------

The semantic specification succinctly describes the features of your task and model. For uploading learnware ``zip`` package, the user need to prepare the semantic specification. Here is an example of a "Table Data" for a "Classification Task":

.. code-block:: python

    from learnware.specification import generate_semantic_spec

    # Prepare input description when data_type="Table"
    input_description = {
        "Dimension": 5,
        "Description": {
            "0": "age",
            "1": "weight",
            "2": "body length",
            "3": "animal type",
            "4": "claw length"
        },
    }

    # Prepare output description when task_type in ["Classification", "Regression"]
    output_description = {
        "Dimension": 3,
        "Description": {
            "0": "cat",
            "1": "dog",
            "2": "bird",
        },
    }

    # Create semantic specification
    semantic_spec = generate_semantic_spec(
        name="learnware_example",
        description="Just an example for uploading learnware",
        data_type="Table",
        task_type="Classification",
        library_type="Scikit-learn",
        scenarios=["Business", "Financial"],
        input_description=input_description,
        output_description=output_description,
    )

For more details, please refer to :ref:`semantic specification<components/spec:Semantic Specification>`, 

Uploading
--------------

you can effortlessly upload your learnware to the ``Learnware Market`` as follows.

.. code-block:: python

    from learnware.market import BaseChecker
    from learnware.market import instantiate_learnware_market

    # instantiate a demo market
    demo_market = instantiate_learnware_market(market_id="demo", name="hetero", rebuild=True) 

    # upload the learnware into the market
    learnware_id, learnware_status = demo_market.add_learnware(zip_path, semantic_spec) 
    
    # assert whether the learnware passed the check and was uploaded successfully.
    assert learnware_status != BaseChecker.INVALID_LEARNWARE, "Insert learnware failed!"

Here, ``zip_path`` refers to the directory of your learnware ``zip`` package. ``learnware_id`` indicates the id assigned by ``Learnware Market``, and the ``learnware_status`` indicates the check status for learnware.

.. note:: 
    The learnware ``zip`` package uploaded into ``LearnwareMarket`` will be checked semantically and statistically, and ``add_learnware`` will return the concrete check status. The check status ``BaseChecker.INVALID_LEARNWARE`` indicates the learnware did not pass the check. For more details about learnware checker, please refer to `Learnware Market <../components/market.html#easy-checker>`

Remove Learnware
==================

As administrators of the ``Learnware Market``, it's crucial to remove learnwares that exhibit suspicious uploading motives.
Once you have the necessary permissions and approvals, you can use the following code to remove a learnware 
from the ``Learnware Market``:

.. code-block:: python

    easy_market.delete_learnware(learnware_id)

Here,  ``learnware_id`` refers to the market ID of the learnware to be removed.
