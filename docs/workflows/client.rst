============================================================
Learnware Client
============================================================


Introduction
====================

``Learnware Client`` is a python api that provides a convenient interface for interacting with the system. You can easily use the client to upload, download and search learnwares.


Prepare access token
====================

Before using the ``Learnware Client``, you'll need to obtain a token from the `official website <https://www.lamda.nju.edu.cn/learnware/>`_. Just login to the website and click ``Client Token`` tab in the user center.


How to Use Client
============================


Initialize a Learware Client
-------------------------------


.. code-block:: python
    
    from learnware.client import LearnwareClient, SemanticSpecificationKey

    # Login to Beiming system
    client = LearnwareClient()
    client.login(email="your email", token="your token")

Where email is the registered mailbox of the system and token is the token obtained in the previous section.

Upload Leanware
-------------------------------

Before uploading a learnware, you'll need to prepare the semantic specification of your learnware. Let's take the classification task for tabular data as an example. You can create a semantic specification by a helper function ``create_semantic_specification``.

.. code-block:: python

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
    semantic_spec = client.create_semantic_specification(
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
    
Ensure that the input parameters for the semantic specification fall within the specified ranges provided by ``client.list_semantic_specification_values(key)``:

* "data_type" must be within the range of ``key=SemanticSpecificationKey.DATA_TYPE``.
* "task_type" must be within the range of ``key=SemanticSpecificationKey.TASK_TYPE``.
* "library_type" must be within the range of ``key=SemanticSpecificationKey.LIBRARY_TYPE``.
* "scenarios" must be a subset of ``key=SemanticSpecificationKey.SENARIOS``.
* "license" must be a subset of ``key=SemanticSpecificationKey.LICENSE``.
* When "data_type" is set to "Table", it is necessary to provide "input_description".
* When "task_type" is either "Classification" or "Regression", it is necessary to provide "output_description".

Finally, the semantic specification and the zip package path of the learnware were filled in to upload the learnware.

Remember to verify the learnware before uploading it, as shown in the following code example:

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

After uploading the learnware successfully, you can see it in ``My Learnware``, the background will check it. Click on the learnware, which can be viewed in the ``Verify Status``. After the check passes, the Unverified tag of the learnware will disappear, and the uploaded learnware will appear in the system.

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

    # Replace with the actual learnware ID
    learnware_id = "123456789"

    # Delete the specified learnware
    client.delete_learnware(learnware_id)


Semantic Specification Search
-------------------------------

You can search the learnware in the system through the semantic specification, and all the learnware conforming to the semantic specification will be returned through the API. For example, the following code will give you all the learnware in the system whose task type is classified:

.. code-block:: python

    from learnware.market import BaseUserInfo

    user_semantic = client.create_semantic_specification(
        task_type="Classification"
    )
    user_info = BaseUserInfo(semantic_spec=user_semantic)
    learnware_list = client.search_learnware(user_info, page_size=None)
    

Statistical Specification Search
---------------------------------

You can also search the learnware in the system through the statistical specification, and all the learnware with similar distribution will be returned through the API. Using the ``generate_stat_spec`` function mentioned above, you can easily get the ``stat_spec`` for your current task, and then get the learnware that meets the statistical specification for the same type of data in the system by using the following code:

.. code-block:: python

    user_info = BaseUserInfo(stat_info={stat_spec.type: stat_spec})
    learnware_list = client.search_learnware(user_info, page_size=None)


Combine Semantic and Statistical Search
----------------------------------------
By combining statistical and semantic specifications, you can perform more detailed searches, such as the following code that searches tabular data for pieces of learnware that satisfy your semantic specifications:

.. code-block:: python

    user_semantic = client.create_semantic_specification(
        task_type="Classification",
        scenarios=["Business"],
    )
    rkme_table = generate_stat_spec(type="table", X=train_x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={rkme_table.type: rkme_table}
    )
    learnware_list = client.search_learnware(user_info, page_size=None)

Heterogeneous Table Search
----------------------------------------
When you provide a statistical specification for tabular data, the task type is "Classification" or "Regression", and your semantic specification includes descriptions for each dimension, the system will automatically enable heterogeneous table search. It won't only search in the tabular learnwares with same dimensions. The following code will perform heterogeneous table search through the API:

.. code-block:: python

    input_description = {
        "Dimension": 2,
        "Description": {
            "0": "leaf width",
            "1": "leaf length",
        },
    }
    user_semantic = client.create_semantic_specification(
        task_type="Classification",
        scenarios=["Business"],
        input_description=input_description,
    )
    rkme_table = generate_stat_spec(type="table", X=train_x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={rkme_table.type: rkme_table}
    )
    learnware_list = client.search_learnware(user_info)


Download and Use Learnware
-------------------------------
When the search is complete, you can download the learnware and configure the environment through the following code:

.. code-block:: python

    for temp_learnware in learnware_list:
        learnware_id = temp_learnware["learnware_id"]

        # you can use the learnware to make prediction now
        learnware = client.load_learnware(
            learnware_id=learnware_id, runnable_option="conda"
        )