============================================================
Learnware Client
============================================================


Introduction
==================== 

``Learnware Client`` is a python api that provides a convenient interface for interacting with the official market. You can easily use the client to upload, download and search learnwares.


Installation
====================

``Learnware Client`` is contained in the ``learnware`` package. You can install it using pip:

.. code-block:: bash

    pip install learnware


Prepare access token
====================

Before using the ``Learnware Client``, you'll need to obtain a token from the `official website <https://www.lamda.nju.edu.cn/learnware/>`_. Just login to the website and click "client token" tab in the user center.


Use Client
============================


Initialize a Learware Client
-------------------------------


.. code-block:: python
    
    import learnware
    from learnware.client import LearnwareClient

    client = LearnwareClient()

    # login to official market
    client.login(email="your email", token="your token")


Upload Leanware
-------------------------------

Before uploading a learnware, you'll need to prepare the semantic specification of your learnware. You can create a semantic specification by a helper function ``create_semantic_specification``.

.. code-block:: python

    input_description = {
        "Dimension": 16,
        {
            "Description": {
                "0": "gender",
                "1": "age",
                "2": "f2",
                "5": "f5"
            }            
        }
    }
    output_description = {
        "Dimension": 3,
        "Description": {
            "0": "the probability of being a cat",
            "1": "the probability of being a dog",
            "2": "the probability of being a bird"
        }
    }
    semantic_spec = client.create_semantic_specification(
        name="mylearnware1", 
        description="this is my learnware", 
        data_type="Table", 
        task_type="Classification", 
        library_type="Scikit-learn", 
        senarioes=["Business", "Financial"],
        input_description, output_description)
    # data_type, task_type, library_type, senarioes are enums, you can find possible values in `learnware.C`
    

After defining the semantic specification, 
you can upload your learnware using ``upload_learnware`` function:
    
.. code-block:: python
    
    learnware_id = client.upload_learnware(
        semantic_spec=semantic_spec, 
        zip_path="path to your learnware zipfile")

Here, ``zip_path`` is the local path of your learnware zipfile.


Semantic Specification Search
-------------------------------

You can search learnwares in official market using semantic specification. All the learnwares that match the semantic specification will be returned by the api. For example, the code below searches learnwares with `Table` data type:

.. code-block:: python

    semantic_spec = client.create_semantic_specification(
        name="", 
        description="", 
        data_type="Table", 
        task_type="", 
        library_type="", 
        senarioes=[],
        input_description={}, output_description={})
    
    specification = learnware.specification.Specification()
    specification.update_semantic_spec(specification)
    learnware_list = client.search_learnware(specification)
    

Statistical Specification Search
---------------------------------

You can search learnware by providing a statistical specification. The statistical specification is a json file that contains the statistical information of your training data. For example, the code below searches learnwares with `RKMETableSpecification`:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.RKMETableSpecification()
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    
    specification = learnware.specification.Specification()
    specification.update_stat_spec(user_spec)

    learnware_list = client.search_learnware(specification)

    # you can view the scores of the searched learnwares
    for learnware in learnware_list:
        print(f'learnware_id: {learnware["learnware_id"]}, score: {learnware["matching"]}')


Combine Semantic and Statistical Search
----------------------------------------
You can provide both semantic and statistical specification to search learnwares. The engine will first filter learnwares by semantic specification and then search by statistical specification. For example, the code below searches learnwares with `Table` data type and `RKMETableSpecification`:

.. code-block:: python

    semantic_spec = client.create_semantic_specification(
        name="", 
        description="", 
        data_type="Table", 
        task_type="", 
        library_type="", 
        senarioes=[],
        input_description={}, output_description={})

    stat_spec = specification.RKMETableSpecification()
    stat_spec.load(os.path.join(unzip_path, "rkme.json"))
    specification = learnware.specification.Specification()
    specification.update_semantic_spec(semantic_spec)
    specification.update_stat_spec(stat_spec)

    learnware_list = client.search_learnware(specification)


Download and Use Learnware
-------------------------------
When you get a learnware id, you can download and initiate the learnware with the following code:

.. code-block:: python

    client.download_learnware(learnware_id, zip_path)
    client.install_environment(zip_path)
    learnware = client.load_learnware(zip_path)
    # you can use the learnware to make prediction now




