============================================================
Learnwares Search
============================================================

``Learnware Searcher`` is a key component of ``Learnware Market`` that identifies and recommends helpful learnwares to users according to their ``UserInfo``. Based on whether the returned learnware dimensions are consistent with user tasks, the searchers can be divided into two categories: homogeneous searchers and heterogeneous searchers.


Homo Search
======================

The homogeneous search of helpful learnwares is based on the user information, and can be divided into two stages: semantic specification search and statistical specification search.

User information
-------------------------------
``BaseUserInfo`` is a ``Python API`` for users to provide enough information to identify helpful learnwares.
When initializing ``BaseUserInfo``, three optional information can be provided: ``id``, ``semantic_spec`` and ``stat_info``. The generation of these specifications is seen in `Learnware Preparation <./submit>`_.



Semantic Specification Search
-------------------------------
To search for learnwares that fit your task purpose, 
the user could first provide a semantic specification ``user_semantic`` that describes the characteristics of your task.
The Learnware Market will perform a first-stage search based on ``user_semantic``,
identifying potentially helpful leaarnwares whose models solve tasks similar to your requirements. There are two types of Semantic Specification Search: ``EasyExactSemanticSearcher`` and ``EasyFuzzSemanticSearcher``. They can be used like:

- `EasyExactSemanticSearcher(learnware_list: List[Learnware], user_info: BaseUserInfo)-> SearchResults`

- `EasyFuzzSemanticSearcher(learnware_list: List[Learnware], user_info: BaseUserInfo)-> SearchResults`

In these two searchers, each learnware in the ``learnware_list`` is compared with ``user_info`` according to their ``semantic_spec``, and added to the search result if mathched. Two semantic_spec are matched when all the key words are matched or empty in ``user_info``. Different keys have different matching rules:

- For keys ``Data``, ``Task``, ``Library`` and ``license``, two ``semantic_spec`` keys are matched only if these values(only one value for each key) of learnware ``semantic_spec`` exists in values(may be muliple values for one key) of user ``semantic_spec``.

- For the key ``Scenario``, two ``semantic_spec`` keys are matched if their values have nonempty intersections.

- For keys ``Name`` and ``Description``, the values are strings and case is ignored;

    - In ``EasyExactSemanticSearcher``, two ``semantic_spec`` keys are matched if these values of learnware ``semantic_spec`` is a substring of user ``semantic_spec``.

    - In ``EasyFuzzSemanticSearcher``, first the exact semantic searcher is conducted like ``EasyExactSemanticSearcher``. If the result is empty, the fuzz semantic searcher is activated: the ``learnware_list`` is sorted according to the fuzz score function ``fuzz.partial_ratio`` in ``rapidfuzz``.

The results are returned storing in ``single_results`` of ``SearchResults``.


Statistical Specification Search
---------------------------------

If you choose to provide your own statistical specification ``stat_info``, 
the Learnware Market can perform a more accurate leanware selection using ``EasyStatSearcher``. 

- `EasyStatSearcher(
        learnware_list: List[Learnware],
        user_info: BaseUserInfo,
        max_search_num: int = 5, 
        search_method: str = "greedy",) 
        -> SearchResults`
    
    - It searches for helpful learnwares from ``learnware_list`` based on the ``stat_info`` in ``user_info``.
  
    - The result ``SingleSearchItem`` and ``MultipleSearchItem`` are both stored in ``SearchResults``. In ``SingleSearchItem``, it searches for single learnwares that could solve the user task; scores are also provided to represent the fitness of each single learnware and user task. In ``MultipleSearchItem``, it searches for a mixture of learnwares that could solve the user task better; the mixture learnware list and a score for the mixture is returned.

    - The parameter ``search_method`` provides two choice of search strategies for mixture learnwares: ``greedy`` and ``auto``. For the search method ``greedy``, each time it chooses a learnware to make their mixture closer to the user's ``stat_info``; for the search method ``auto``, it directly calculates a best mixture weight for the ``learnware_list``.

    - For single learnware search, we only return the learnwares with score larger than 0.6; For multiple learnware search, the parameter ``max_search_num`` specifies the maximum length of the returned  mixture learnware list. 

Semantic and Statistical Specification Search
-------------------------------

The semantic specification search and statistical specification search have been Has been integrated into the same interface ``EasySearcher``. 

Hetero Search
======================