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