.. _api:
================================
API Reference
================================

Here you can find high-level interfaces in the ``learnware`` package.

Market
====================

.. autoclass:: learnware.market.LearnwareMarket
    :members:

.. autoclass:: learnware.market.BaseUserInfo
    :members:

Organizer
------------------
.. autoclass:: learnware.market.BaseOrganizer
    :members:

.. autoclass:: learnware.market.EasyOrganizer
    :members:

.. autoclass:: learnware.market.HeteroMapTableOrganizer
    :members:

Searcher
------------------
.. autoclass:: learnware.market.BaseSearcher
    :members:

.. autoclass:: learnware.market.EasySearcher
    :members:

.. autoclass:: learnware.market.easy.EasyExactSemanticSearcher
    :members:

.. autoclass:: learnware.market.easy.EasyFuzzSemanticSearcher
    :members:

.. autoclass:: learnware.market.easy.EasyStatSearcher
    :members:

.. autoclass:: learnware.market.HeteroSearcher
    :members:

Checker
------------------

.. autoclass:: learnware.market.BaseChecker
    :members:

.. autoclass:: learnware.market.EasySemanticChecker
    :members:

.. autoclass:: learnware.market.EasyStatChecker
    :members:

Learnware
====================

.. autoclass:: learnware.learnware.Learnware
    :members:

Reuser
====================

.. autoclass:: learnware.reuse.BaseReuser
    :members:

Data Independent Reuser
-------------------------

.. autoclass:: learnware.reuse.JobSelectorReuser
    :members:

.. autoclass:: learnware.reuse.AveragingReuser
    :members:

Data Dependent Reuser
-------------------------

.. autoclass:: learnware.reuse.EnsemblePruningReuser
    :members:

.. autoclass:: learnware.reuse.FeatureAugmentReuser
    :members:


Aligned Learnware
--------------------
    
.. autoclass:: learnware.reuse.AlignLearnware
    :members:

.. autoclass:: learnware.reuse.FeatureAlignLearnware
    :members:

.. autoclass:: learnware.reuse.HeteroMapAlignLearnware
    :members:

Specification
====================

.. autoclass:: learnware.specification.Specification
    :members:

.. autoclass:: learnware.specification.BaseStatSpecification
    :members:

Regular Specification
--------------------------

.. autoclass:: learnware.specification.RegularStatSpecification
    :members:

.. autoclass:: learnware.specification.RKMETableSpecification
    :members:

.. autoclass:: learnware.specification.RKMEImageSpecification
    :members:

.. autoclass:: learnware.specification.RKMETextSpecification
    :members:

System Specification
--------------------------

.. autoclass:: learnware.specification.HeteroMapTableSpecification
    :members:

Model
====================


Base Model
--------------
.. autoclass:: learnware.model.BaseModel
    :members:

Container
-------------

.. autoclass:: learnware.client.container.ModelContainer
    :members:

.. autoclass:: learnware.client.container.ModelCondaContainer
    :members:

.. autoclass:: learnware.client.container.ModelDockerContainer
    :members:

.. autoclass:: learnware.client.container.LearnwaresContainer
    :members: