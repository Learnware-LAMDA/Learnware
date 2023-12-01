.. learnware documentation master file, created by
   sphinx-quickstart on Tue Mar 28 22:06:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================================================
``Learnware`` Documentation
============================================================

``Learnware`` is a model sharing platform, which give a basic implementation of the learnware paradigm. A learnware is a well-performed trained machine learning model with a specification that enables it to be adequately identified to reuse according to the requirement of future users who may know nothing about the learnware in advance. The learnware paradigm can solve entangled problems in the current machine learning paradigm, like continual learning and catastrophic forgetting. It also reduces resources for training a well-performed model. 

.. _user_guide:

Document Structure
====================

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 3
   :caption: GETTING STARTED:

   Introduction <start/intro.rst>
   Quick Start <start/quick.rst>
   Installation <start/install.rst>
   Experiments and Examples <start/exp.rst>

.. toctree::
   :maxdepth: 3
   :caption: WORKFLOWS:

   Learnware Preparation and Uploading <workflows/upload.rst>
   Learnware Search <workflows/search.rst>
   Learnware Reuse <workflows/reuse.rst>
   Learnware Client <workflows/client.rst>

.. toctree::
   :maxdepth: 3
   :caption: COMPONENTS:

   Market <components/market.rst>
   Learnware & Reuser <components/learnware.rst>
   Specification <components/spec.rst>
   Model & Container <components/model.rst>

.. toctree::
   :maxdepth: 3
   :caption: ADVANCED TOPICS:

   Anchor Learnware <advanced/anchor.rst>
   Specification Evolvement <advanced/evolve.rst>

.. toctree::
   :maxdepth: 3
   :caption: REFERENCES:

   API <references/api.rst>
   BeimingWu System <references/beiming.rst>
   FAQ <references/FAQ.rst>

.. toctree::
   :maxdepth: 3
   :caption: ABOUTS:

   For Developer <about/dev.rst>
   Changelog <about/changelog.rst>
   About us <about/about.rst>

