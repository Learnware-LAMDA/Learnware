================
Anchor learnware
================

Anchor learnwares are a small fraction of representative learnwares that helps locate user's requirements through user feedback. The learnware market can choose or generate several learnwares as anchor learnwares corresponding to the specification island. If the user does not have sufficient training data for constructing an RKME requirement, the learnware market can send several anchor learnwares to the user. By feeding her own data to these anchor learnwares, some information such as (precision, recall) or other performance indicators, can be generated and returned to the market. These information could help the market identify potentially helpful models, e.g., by identifying models that are far from anchors exhibiting poor performance whereas close to anchors exhibiting relatively better performance in the specification island.

To fulfill the anchor learnware method, you need to implement the following functions in the folder ``Learnware/market/anchor``: 

- First, you should design how the market chooses or generates anchor learnwares. This can be realized by selecting prototype models through functional space clustering, and more interesting designs can be explored. The class ``AnchoredOrganizer`` in ``organizer.py`` is designed for it. The function ``AnchoredOrganizer.update_anchor_learnware_list`` is reserved for choosing or generating anchor learnwares. The functions ``AnchoredOrganizer._update_anchor_learnware`` and ``AnchoredOrganizer._delete_anchor_learnware`` have been completed as auxiliary.

- Second, when a user comes with no RKME(or other statistical) specifications, the market should choose several anchor learnwares and send them to the user. This process is done by ``AnchoredSearcher.search_anchor_learnware`` in ``searcher.py``. Besides the list of anchor learnwares, it also returns an item specifying which performance indicator should the user return. 

- Third, by feeding the user's data to these anchor learnwares, the returned information is calculated and stored in ``AnchoredUserInfo`` in ``user_info.py``. The user should add the anchor learnwares into it using ``AnchoredUserInfo.add_anchor_learnware_ids``, and then fill in performance indicator using ``AnchoredUserInfo.update_stat_info``.

- Fourth, according to the returned information from the user, the market should identify the helpful learnwares for the user. This process is done in ``AnchoredSearcher.search_learnware`` in ``searcher.py``, which returns the recommended comination of helpful learnwares and the list of helpful learnwares.
  
- Fourth, according to the returned information from the user, the market should identify the helpful learnwares for the user. This process is done in ``AnchoredMarket.search_learnware``.



