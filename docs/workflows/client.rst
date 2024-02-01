============================================================
Learnware Client
============================================================


Introduction
====================

``Learnware Client`` is a ``Python API`` that provides a convenient interface for interacting with the ``Beimingwu`` system. You can easily use the client to upload, download, delete, update, and search learnwares.


Prepare access token
====================

Before using the ``Learnware Client``, you'll need to obtain a token from the `official website <https://bmwu.cloud/>`_. Just login to the website and click ``Client Token`` tab in the ``Personal Information``.


How to Use Client
============================


Initialize a Learnware Client
-------------------------------


.. code-block:: python
    
    from learnware.client import LearnwareClient, SemanticSpecificationKey

    # Login to Beiming system
    client = LearnwareClient()
    client.login(email="your email", token="your token")

Where email is the registered mailbox of the system and token is the token obtained in the previous section.

Upload Leanware
-------------------------------

Before uploading a learnware, you'll need to prepare the semantic specification of your learnware. Let's take the classification task for tabular data as an example. You can create a semantic specification by a helper function ``generate_semantic_spec``.

.. code-block:: python

    from learnware.specification import generate_semantics_spec

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
        description="Just a example for uploading a learnware",
        data_type="Table",
        task_type="Classification",
        library_type="Scikit-learn",
        scenarios=["Business", "Financial"],
        license=["Apache-2.0"],
        input_description=input_description,
        output_description=output_description,
    )
    
Please ensure that the input parameters for the semantic specification fall within the specified ranges provided by ``client.list_semantic_specification_values(key)``:

* "data_type" must be within the range of ``key=SemanticSpecificationKey.DATA_TYPE``.
* "task_type" must be within the range of ``key=SemanticSpecificationKey.TASK_TYPE``.
* "library_type" must be within the range of ``key=SemanticSpecificationKey.LIBRARY_TYPE``.
* "scenarios" must be a subset of ``key=SemanticSpecificationKey.SENARIOS``.
* "license" must be within the range of ``key=SemanticSpecificationKey.LICENSE``.
* When "data_type" is set to "Table", it is necessary to provide "input_description".
* When "task_type" is either "Classification" or "Regression", it is necessary to provide "output_description".

Finally, the semantic specification and the zip package path of the learnware were filled in to upload the learnware.

Remember to validate your learnware before uploading it, as shown in the following code example:

.. code-block:: python

    # Prepare your learnware zip file
    zip_path = "your learnware zip"

    # Check your learnware before upload
    client.check_learnware(
        learnware_zip_path=zip_path, semantic_specification=semantic_spec
    )

    # Upload your learnware
    learnware_id = client.upload_learnware(
        learnware_zip_path=zip_path, semantic_specification=semantic_spec
    )

After uploading the learnware successfully, you can see it in ``Personal Information - My Learnware``, the background will check it. Click on the learnware, which can be viewed in the ``Verify Status``. After the check passes, the Unverified tag of the learnware will disappear, and the uploaded learnware will appear in the system.

Update Learnware
-------------------------------

The ``update_learnware`` method is used to update the metadata and content of an existing learnware on the server. You can upload a new semantic specification, or directly upload a new learnware.

.. code-block:: python

    # Replace with the actual learnware ID
    learnware_id = "123456789"

    # Create new semantic specification
    semantic_spec = client.create_semantic_specification(
        name="new learnware name",
        description="new description",
        data_type="Table",
        task_type="Classification",
        library_type="Scikit-learn",
        scenarios=["Computer", "Internet"],
        license=["CC-BY-4.0"],
        input_description=new_input_description,
        output_description=new_output_description,
    )

    # Update metadata without changing the content
    client.update_learnware(learnware_id, semantic_spec)

    # Update metadata and content with a new ZIP file
    updated_zip_path = "/path/to/updated_learnware.zip"
    client.update_learnware(learnware_id, semantic_spec, learnware_zip_path=updated_zip_path)

Delete Learnware
-------------------------------

The ``delete_learnware`` method is used to delete a learnware from the server.

.. code-block:: python

    # Replace with the actual learnware ID to delete
    learnware_id = "123456789"

    # Delete the specified learnware
    client.delete_learnware(learnware_id)


Semantic Specification Search
-------------------------------

You can search for learnware(s) in the system through semantic specifications, and all learnwares that meet the semantic specifications will be returned via the API. For example, the following code retrieves all learnware in the system with a task type of "Classification":

.. code-block:: python

    from learnware.market import BaseUserInfo

    user_semantic = generate_semantic_spec(
        task_type="Classification"
    )
    user_info = BaseUserInfo(semantic_spec=user_semantic)
    search_result = client.search_learnware(user_info)


Statistical Specification Search
---------------------------------

Moreover, you can also search for learnware(s) in the learnware dock system through statistical specifications, and more targeted learnwares for your task will be returned through the API. Using the ``generate_stat_spec`` function mentioned above, you can generate your task's statistical specification ``stat_spec``. Then, you can use the following code to easily obtain suitable learnware(s) identified by the system for your specific task:

.. code-block:: python

    user_info = BaseUserInfo(stat_info={stat_spec.type: stat_spec})
    search_result = client.search_learnware(user_info)


Combine Semantic and Statistical Search
----------------------------------------

By combining both semantic and statistical specifications, you can perform more accurate searches. For instance, the code below demonstrates how to search for learnware(s) in tabular data that satisfy both the semantic and statistical specifications:

.. code-block:: python

    from learnware.specification import generate_stat_spec

    user_semantic = generate_semantic_spec(
        task_type="Classification",
        scenarios=["Business"],
    )
    rkme_table = generate_stat_spec(type="table", X=train_x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={rkme_table.type: rkme_table}
    )
    search_result = client.search_learnware(user_info)


Heterogeneous Table Search
----------------------------------------
For tabular tasks, if the task type is "Classification" or "Regression", and you have provided a statistical specification along with descriptions for each feature dimension in the semantic specification, the system will enable heterogeneous table search. This is designed to support searching models from different feature spaces preliminarily. The following code example shows how to perform a heterogeneous table search via the API:

.. code-block:: python

    input_description = {
        "Dimension": 2,
        "Description": {
            "0": "leaf width",
            "1": "leaf length",
        },
    }
    user_semantic = generate_semantic_spec(
        task_type="Classification",
        scenarios=["Business"],
        input_description=input_description,
    )
    rkme_table = generate_stat_spec(type="table", X=train_x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={rkme_table.type: rkme_table}
    )
    search_result = client.search_learnware(user_info)


Download and Use Learnware
-------------------------------
After the learnware search is completed, you can locally load and use the learnwares through the learnware IDs in ``search_result``, as shown in the following example:

.. code-block:: python

    learnware_id = search_result["single"]["learnware_ids"][0]
    learnware = client.load_learnware(
        learnware_id=learnware_id, runnable_option="conda"
    )
    # test_x is the user's data for prediction
    predict_y = learnware.predict(test_x)