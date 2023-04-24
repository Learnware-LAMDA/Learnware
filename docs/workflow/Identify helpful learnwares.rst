============================================================
Identify Helpful Learnwares
============================================================

Semantic Specification Search
-------------------------------
To search for learnwares that fit your task purpose, 
you should first provide a semantic specification ``user_semantic`` that describes the characteristics of your task.
The Learnware Market will perform a first-stage search based on ``user_semantic``,
identifying potentially helpful leranwares whose models solve tasks similar to your requirements. 

.. code-block:: python

    # construct user_info which includes semantic specification for searching learnware
    user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

    # search_learnware performs semantic specification search if user_info doesn't include a statistical specification
    _, single_learnware_list, _ = easy_market.search_learnware(user_info) 

    # single_learnware_list is the learnware list by semantic specification searching
    print(single_learnware_list)

In semantic specification search, we go through all learnwares in the market to compare their semantic specifications with the user's one, and return all the learnwares that pass through the comparation. When comparing two learnwares' semantic specifications, we design different ways for different semantic keys:

- For semantic keys with type 'Class', they are matched only if they have the same value.

- For semantic keys with type 'Tag', they are matched only if they have nonempty intersections.

- For the user's input in the search box, it matchs with a learnware's semantic specification only if it's a substring of its 'Name' or 'Description'. All the strings are converted to the lower case before matching.

- When a key value is missing, it will not participate in the match. The user could upload no semantic specifications if he wants.

Statistical Specification Search
---------------------------------

If you choose to provide your own statistical specification file ``stat.json``, 
the Learnware Market can perform a more accurate leanware selection from 
the learnwares returned by the previous step. This second-stage search is based on statistical information and returns one or more learnwares that are most likely to be helpful for your task. 

For example, the following code is designed to work with Reduced Kernel Mean Embedding (RKME) as a statistical specification:

.. code-block:: python

    import learnware.specification as specification

    user_spec = specification.rkme.RKMEStatSpecification()
    user_spec.load(os.path.join(unzip_path, "rkme.json"))
    user_info = BaseUserInfo(
        id="user", semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": user_spec}
    )
    (sorted_score_list, single_learnware_list,
        mixture_score, mixture_learnware_list) = easy_market.search_learnware(user_info)

    # sorted_score_list is the learnware scores based on MMD distances, sorted in descending order
    print(sorted_score_list) 

    # single_learnware_list is the learnwares sorted in descending order based on their scores
    print(single_learnware_list)

    # mixture_learnware_list is the learnwares whose mixture is helpful for your task
    print(mixture_learnware_list) 

    # mixture_score is the score of the mixture of learnwares
    print(mixture_score)

The statistical specification search is done in the following way.
We first filter by the dimension of RKME specifications; only those with the same dimension with the user's will enter the subsequent stage.

The single_learnware_list is calculated using the distances between two RKMEs. The greater the distance from the user's RKME, the lower the score is. The learnwares with very low scores will not be returned, unless the number of matching leanwares is smaller than a set threshold.

The mixture_learnware_list is calculated in a greedy way. Each time we choose a learnware to make their mixture closer to the user's RKME. Specifically, each time we go through all the left learnwares to find the one whose combination with chosen learnwares could minimize the distance between their mixture's RKME and the user's RKME. The mixture weight is calculated by minimizing the RKME distance, which is solved by quadratic programming. If the distance become larger or the number of chosen learnwares reaches a threshold, the process will end and the chosen learnware and weight list will return.