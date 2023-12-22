============================================================
Learnwares Search
============================================================

``Learnware Searcher`` is a key component of ``Learnware Market`` that identifies and recommends helpful learnwares to users according to their ``UserInfo``. Based on whether the returned learnware dimensions are consistent with user tasks, the searchers can be divided into two categories: homogeneous searchers and heterogeneous searchers. 

All the searchers are implemented as a subclass of ``BaseSearcher``. When initializing, you should assign a ``organizer`` to it. The introduction of ``organizer`` is shown in `COMPONENTS: Market - Framework <../components/market.html>`_. Then these searchers can be called with ``UserInfo`` and return ``SearchResults``.

Homo Search
======================

The homogeneous search of helpful learnwares can be divided into two stages: semantic specification search and statistical specification search. Both of them needs ``BaseUserInfo`` as input. The following codes shows how to use the searcher to search for helpful learnwares from a market ``easy_market`` for a user. The introduction of ``EasyMarket`` is in `COMPONENTS: Market <../components/market.html>`_.

.. code-block:: python

    # generate BaseUserInfo(semantic_spec + stat_info)
    user_semantic = {
        "Data": {"Values": ["Table"], "Type": "Class"},
        "Task": {"Values": ["Regression"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "", "Type": "String"},
        "Input": {"Dimension": 82, "Description": {},},
        "Output": {"Dimension": 1, "Description": {},}, 
        "License": {"Values": ["MIT"], "Type": "Class"},
    }
    user_spec = generate_rkme_table_spec(X=x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, 
        stat_info={"RKMETableSpecification": user_spec}
    )

    # search the market for the user
    search_result = easy_market.search_learnware(user_info)

    # search result: single_result
    single_result = search_result.get_single_results()
    print(f"single model num: {len(single_result)}, 
        max_score: {single_result[0].score}, 
        min_score: {single_result[-1].score}"
    )
    
    # search result: multiple_result
    multiple_result = search_result.get_multiple_results()
    mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
    print(f"mixture_score: {multiple_result[0].score}, mixture_learnwares: {mixture_id}")

Hetero Search
======================

For table-based user tasks, 
homogeneous searchers like ``EasySearcher`` fail to recommend learnwares when no table learnware matches the user task's feature dimension, returning empty results.
To enhance functionality, ``learnware`` package includes the heterogeneous learnware search feature, whose processions is as follows: 

- Learnware markets such as ``Hetero Market`` integrate different specification islands into a unified "specification world" by assigning system-level specifications to all learnwares. This allows heterogeneous searchers like ``HeteroSearcher`` to find helpful learnwares from all available table learnwares.
- Searchers assign system-level specifications to users based on ``UserInfo``'s statistical specification, using methods provided by corresponding organizers. In ``Hetero Market``, for example, ``HeteroOrganizer.generate_hetero_map_spec`` generates system-level specifications for users.
- Finally searchers conduct statistical specification search across the "specification world". User's system-level specification will guide the searcher in pinpointing helpful heterogeneous learnwares.

To activate heterogeneous learnware search, ``UserInfo`` should contain both semantic and statistical specifications. What's more, the semantic specification should meet the following requirements: 

- The task type should be ``Classification`` or ``Regression``.
- The data type should be ``Table``.
- It should include description for at least one feature dimension.
- The feature dimension stated here should match with the feature dimension in the statistical specification.

The following codes shows how to search for helpful heterogeneous learnwares from a market 
``hetero_market`` for a user. The introduction of ``HeteroMarket`` is in `COMPONENTS: Hetero Market <../components/market.html#hetero-market>`_.

.. code-block:: python

  # initiate a Hetero Market
  hetero_market = initiate_learnware_market(market_id="test_hetero", name="hetero")
  
  # user_semantic should meet the above requirements
  input_description = {
      "Dimension": 2,
      "Description": {
          "0": "leaf width",
          "1": "leaf length",
      },
  }
  user_semantic = generate_semantic_spec(
      data_type="table",
      task_type="Classification",
      scenarios=["Business"],
      input_description=input_description,
  )
  user_spec = generate_stat_spec(type="table", X=train_x)
  user_info = BaseUserInfo(
      semantic_spec=user_semantic,
      stat_info={user_spec.type: user_spec}
  )

  # search for heterogeneous learnwares in hetero_market
  search_result = hetero_market.search_learnware(user_info)