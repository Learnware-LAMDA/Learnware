================
Introduction
================

``Learnware`` is a model sharing platform, which give a basic implementation of the learnware paradigm. A learnware is a well-performed trained machine learning model with a specification that enables it to be adequately identified to reuse according to the requirement of future users who may know nothing about the learnware in advance. The learnware paradigm can solve entangled problems in the current machine learning paradigm, like continual learning and catastrophic forgetting. It also reduces resources for training a well-performed model.


Motivation
=================

Machine learning, especially the prevailing big model paradigm, has achieved great success in natural language processing and computer vision applications. However, it still faces challenges such as the requirement of a large amount of labeled training data, difficulty in adapting to changing environments, and catastrophic forgetting when refining trained models incrementally. These big models, while useful in their targeted tasks, often fail to address the above issues and struggle to generalize beyond their specific purposes.

To better address the entangled issues in machine learning, we should consider the following aspects:

+------------------------------------------------------------------------------------+
| Aspect                                                                             |
+====================================================================================+
| 1. Investigate techniques that address multiple challenges simultaneously,         |
|    recognizing that these issues are often intertwined in real-world applications. |
+------------------------------------------------------------------------------------+
| 2. Explore paradigms like learnware, which offers the possibility of               |
|    systematically reusing small models for tasks beyond their original purposes,   |
|    reducing the need for users to build models from scratch.                       |
+------------------------------------------------------------------------------------+
| 3. Develop solutions that enable ordinary users to create well-performing models   |
|    without requiring proficient training skills.                                   |
+------------------------------------------------------------------------------------+
| 4. Address data privacy and proprietary concerns to facilitate experience          |
|    sharing among different users while respecting confidentiality.                 |
+------------------------------------------------------------------------------------+
| 5. Adapt to the constraints of big data applications, where it may be              |
|    unaffordable or infeasible to hold all data for multiple passes of scanning.    |
+------------------------------------------------------------------------------------+
| 6. Consider the environmental impact of training large models, as their carbon     |
|    emissions pose a threat to our environment.                                     |
+------------------------------------------------------------------------------------+

By considering these factors, we can develop a more comprehensive framework for tackling the complex challenges in machine learning, moving beyond the limitations of the big model paradigm, called Learnware.



Framework
=======================


The learnware paradigm introduces the concept of a well-performed, trained machine learning model with a specification that allows future users, who have no prior knowledge of the learnware, to reuse it based on their requirements.

Developers or owners of trained machine learning models can submit their models to a learnware market. If accepted, the market assigns a specification to the model and accommodates it. The learnware market could host thousands or millions of well-performed models from different developers, for various tasks, using diverse data, and optimizing different objectives.

Instead of building a model from scratch, users can submit their requirements to the learnware market, which then identifies and deploys helpful learnware(s) based on the specifications. Users can apply the learnware directly, adapt it using their data, or exploit it in other ways to improve their model. This process is more efficient and less expensive than building a model from scratch.

Benefits of the Learnware Paradigm
==============================================

+-----------------------+-----------------------------------------------------------------------------------------------+
| Benefit               | Description                                                                                   |
+=======================+===============================================================================================+
| Lack of training data | Strong models can be built with small data by adapting well-performed learnwares.             |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Lack of training      | Ordinary users can obtain strong models by leveraging well-performed learnwares instead of    |
| skills                | building models from scratch.                                                                 |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Catastrophic          | Accepted learnwares are always stored in the learnware market, retaining old knowledge.       |
| forgetting            |                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Continual learning    | The learnware market continually enriches its knowledge with constant submissions of          |
|                       | well-performed learnwares.                                                                    |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Data privacy/         | Developers only submit models, not data, preserving data privacy/proprietary.                 |
| proprietary           |                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Unplanned tasks       | Open to all legal developers, the learnware market can accommodate helpful learnwares for     |
|                       | various tasks.                                                                                |
+-----------------------+-----------------------------------------------------------------------------------------------+
| Carbon emission       | Assembling small models may offer good-enough performance, reducing interest in training      |
|                       | large models and the carbon footprint.                                                        |
+-----------------------+-----------------------------------------------------------------------------------------------+

Challenges and Future Work
==============================================

Although the learnware proposal shows promise, much work remains to make it a reality. The next sections will present some of the progress made so far.




