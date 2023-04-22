================
Background
================

Machine learning, especially the prevailing big model paradigm, has achieved great success in natural language processing and computer vision applications. However, it still faces challenges such as the requirement of a large amount of labeled training data, difficulty in adapting to changing environments, and catastrophic forgetting when refining trained models incrementally. These big models, while useful in their targeted tasks, often fail to address the above issues and struggle to generalize beyond their specific purposes.

To better address the entangled issues in machine learning, we should consider the following aspects:

1. Investigate techniques that address multiple challenges simultaneously, recognizing that these issues are often intertwined in real-world applications.
2. Explore paradigms like learnware, which offers the possibility of systematically reusing small models for tasks beyond their original purposes, reducing the need for users to build models from scratch.
3. Develop solutions that enable ordinary users to create well-performing models without requiring proficient training skills.
4. Address data privacy and proprietary concerns to facilitate experience sharing among different users while respecting confidentiality.
5. Adapt to the constraints of big data applications, where it may be unaffordable or infeasible to hold all data for multiple passes of scanning.
6. Consider the environmental impact of training large models, as their carbon emissions pose a threat to our environment.

By considering these factors, we can develop a more comprehensive framework for tackling the complex challenges in machine learning, moving beyond the limitations of the big model paradigm, called Learnware.

.. contents:: Table of Contents

================
Introduction
================

Learnware is an AI-oriented paradigm aimed at empowering users to create versatile and efficient machine learning models. By reusing small models for tasks beyond their original purposes, Learnware enables users to experiment with and develop better strategies without building models from scratch.


================
Framework
================
The Learnware framework consists of the following components, designed as loosely-coupled modules that can be used independently:

Name                              Description
--------------------------------  --------------------------------------------------------------------------------
Infrastructure layer              Provides underlying support for machine learning research, including high-performance data management and retrieval and flexible model training control.
Learning Framework layer          Contains learnable Forecast Models and Trading Agents that leverage reinforcement learning and supervised learning paradigms, integrated with the Workflow layer for information extraction and environment creation.
Workflow layer                    Covers the entire machine learning workflow, supporting both supervised-learning-based strategies and RL-based strategies, including Information Extractor, Forecast Model, Decision Generator, and Execution Environment.
Interface layer                   Presents a user-friendly interface for the underlying system, providing detailed analysis reports of forecasting signals, models, and execution results.

The Learnware framework may be intimidating for new users, but it accurately captures the details of the design. Users new to Learnware can initially skip it and revisit it later.

.. note::
   The modules with hand-drawn style are under development and will be released in the future.
   The modules with dashed borders are highly user-customizable and extendible.

.. tip::
   The framework image is created with https://draw.io/
.. contents:: Table of Contents
