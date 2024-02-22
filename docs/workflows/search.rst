============================================================
Learnwares Search
============================================================

``Learnware Searcher`` is a key module within the ``Learnware Market`` that identifies and recommends helpful learnwares to users according to their user information. The ``learnware`` package currently provide two types of learnware searchers: 

- homogeneous searchers conduct homogeneous learnware identification and return helpful learnware(s) within the same feature space as the user's task;
- heterogenous searchers preliminarily support heterogenous learnware identification for tabular tasks, which broaden the search scope and return targeted learnware(s) from different feature spaces.

All the searchers are implemented as a subclass of ``BaseSearcher``. When initializing, an ``organizer`` should be assigned to it. 
The introduction of ``organizer`` is shown in `COMPONENTS: Market - Framework <../components/market.html>`_. 
Then, these searchers can be invoked with user information provided in ``BaseUserInfo``, and they will return ``SearchResults`` containing identification results.

Homo Search
======================

The homogeneous search of helpful learnwares can be divided into two stages: semantic specification search and statistical specification search. Both of them needs ``BaseUserInfo`` as input. 
The following codes shows how to use the searcher to search for helpful learnwares from a market ``easy_market`` for a user. 
The introduction of ``EasyMarket`` is in `COMPONENTS: Market <../components/market.html>`_.

.. code-block:: python

    from learnware.market import BaseUserInfo, instantiate_learnware_market
    from learnware.specification import generate_semantic_spec, generate_stat_spec

    easy_market = instantiate_learnware_market(market_id="demo", name="easy", rebuild=True)

    # generate BaseUserInfo(semantic_spec + stat_info)
    user_semantic = generate_semantic_spec(
        task_type="Classification",
        scenarios=["Business"],
    )
    rkme_table = generate_stat_spec(type="table", X=train_x)
    user_info = BaseUserInfo(
        semantic_spec=user_semantic, stat_info={rkme_table.type: rkme_table}
    )
    search_result = easy_market.search_learnware(user_info)

In the above code, ``search_result`` is of type dict, with the following specific structure (``"single"`` and ``"multiple"`` correspond to the search results for a single learnware and multiple learnwares, respectively):

.. code-block:: python

    search_result = {
        "single": {
            "learnware_ids": List[str],
            "semantic_specifications": List[dict],
            "matching": List[float],
        },
        "multiple": {
            "learnware_ids": List[str],
            "semantic_specifications": List[dict],
            "matching": float,
        },
    }

Hetero Search
======================

For tabular tasks, homogeneous searchers like ``EasySearcher`` may fail to recommend learnwares if no table learnware shares the same feature space as the user's task, resulting in empty returns. The ``learnware`` package preliminarily supports the search of learnwares from different feature spaces through heterogeneous searchers. The process is as follows:

- Learnware markets such as ``Hetero Market`` integrate different tabular specification islands into a unified "specification world" by generating new system specifications for learnwares. This allows heterogeneous searchers like ``HeteroSearcher`` to recommend tabular learnwares from the entire learnware collection.
- Based on their statistical specifications, users receive new specifications assigned by searchers, which employ methods from the respective organizers. For instance, in ``Hetero Market``, ``HeteroOrganizer.generate_hetero_map_spec`` is used to generate new specifications for users.
- Finally searchers conduct statistical specification search across the unified "specification world" based on users' new specifications and return potentially targeted heterogeneous learnwares.

To activate heterogeneous learnware search, ``UserInfo`` needs to include both semantic and statistical specifications. Furthermore, the semantic specification should meet the following requirements: 

- The task type should be ``Classification`` or ``Regression``.
- The data type should be ``Table``.
- There should be a description for at least one feature dimension.
- The feature dimension mentioned here must align with that in the statistical specification.

The code below demonstrates how to search for potentially useful heterogeneous learnwares from a market ``hetero_market`` for a user. 
For more information about ``HeteroMarket``, see `COMPONENTS: Hetero Market <../components/market.html#hetero-market>`_.


.. code-block:: python

  # initiate a Hetero Market
  hetero_market = initiate_learnware_market(market_id="demo", name="hetero", rebuild=True)
  
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