============================================================
Learnwares Search
============================================================

``Learnware Searcher`` is a key component of ``Learnware Market`` that identifies and recommends helpful learnwares to users according to their ``UserInfo``. Based on whether the returned learnware dimensions are consistent with user tasks, the searchers can be divided into two categories: homogeneous searchers and heterogeneous searchers. 

All the searchers are implemented as a subclass of ``BaseSearcher``. When initializing, you should assign a ``organizer`` to it. The introduction of ``organizer`` is shown in `COMPONENTS: Market - Framework <../components/market.html>`_. Then these searchers can be called with ``UserInfo`` and return ``SearchResults``.


Homo Search
======================

The homogeneous search of helpful learnwares can be divided into two stages: semantic specification search and statistical specification search. Both of them needs ``BaseUserInfo`` as input.


Hetero Search
======================