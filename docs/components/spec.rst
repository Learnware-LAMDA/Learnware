.. _spec:
================================
Specification
================================

Learnware specification is the core component of the learnware paradigm, linking all processes about learnwares, including uploading, organizing, searching, deploying and reusing. 

In this section, we will introduce the concept and design of learnware specification in the ``learnware`` package.
We will then explore ``regular specification``\ s tailored for different data types such as tables, images and texts.
Lastly, we cover a ``system specification`` specifically assigned to table learnwares by the learnware market, aimed at accommodating all available table learnwares into a unified "specification world" despite their heterogeneity.

Concepts & Types
==================

The learnware specification describes the model's specialty and utility in a certain format, allowing the model to be identified and reused by future users who may have no prior knowledge of the learnware.
The ``learnware`` package employs a highly extensible specification design, which consists of two parts:

- **Semantic specification** describes the model's type and functionality through a set of descriptions and tags. Learnwares with similar semantic specifications reside in the same specification island
- **Statistical specification** characterizes the statistical information contained in the model using various machine learning techniques. It plays a crucial role in locating the appropriate place for the model within the specification island.

When searching in the learnware market, the system first locates specification islands based on the semantic specification of the user's task, 
then pinpoints highly beneficial learnwares on theses islands based on the statistical specification of the user's task.

Statistical Specification
---------------------------

We employ the ``Reduced Kernel Mean Embedding (RKME) Specification`` as the foundation for implementing statistical specification for diverse data types, 
with adjustments made according to the characteristics of each data type. 
The RKME specification is a recent development in learnware specification design, which represents the distribution of a model's training data in a privacy-preserving manner.

Within the ``learnware`` package, you'll find two types of statistical specifications: ``regular specification`` and ``system specification``. The former is generated locally
by users to express their model's statistical information, while the latter is assigned by the learnware market to accommodate and organize heterogeneous learnwares. 

Semantic Specification
-----------------------

The semantic specification consists of a "dict" structure that includes keywords "Data", "Task", "Library", "Scenario", "License", "Description", and "Name". 
In the case of table learnwares, users should additionally provide descriptions for each feature dimension and output dimension through the "Input" and "Output" keywords.


Regular Specification
======================================

The ``learnware`` package provides a unified interface, ``generate_stat_spec``, for generating ``regular specification``\ s across different data types. 
Users can use the training data ``train_x`` (supported types include numpy.ndarray, pandas.DataFrame, and torch.Tensor) as input to generate the ``regular specification`` of the model,
as shown in the following code:

.. code:: python

   for learnware.specification import generate_stat_spec

   data_type = "table" # supported data types: ["table", "image", "text"]
   regular_spec = generate_stat_spec(type=data_type, x=train_x)
   regular_spec.save("stat.json")

It's worth noting that the above code only runs on user's local computer and does not interact with any cloud servers or leak any local private data.

.. note:: 

   In cases where the model's training data is too large, causing the above code to fail, you can consider sampling the training data to ensure it's of a suitable size before proceeding with reduction generation.

Table Specification
--------------------------

The ``regular specification`` for tabular learnware is essentially the RKME specification of the model's training table data. No additional adjustment is needed.

Image Specification
--------------------------

Image data lives in a higher dimensional space than other data types. Unlike lower dimensional spaces, metrics defined based on Euclidean distances (or similar distances) will fail in higher dimensional spaces. This means that measuring the similarity between image samples becomes difficult. 

To address these issues, we use the Neural Tangent Kernel (NTK) based on Convolutional Neural Networks (CNN) to measure the similarity of image samples.  As we all know, CNN has greatly advanced the field of computer vision and is still a mainstream deep learning technique. 

Usage & Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this part, we show that how to generate Image Specification for the training set of the CIFAR-10 dataset. 
Note that the Image Specification is generated on a subset of the CIFAR-10 dataset with ``generate_rkme_image_spec``. 
Then, it is saved to file "cifar10.json" using ``spec.save``. 

In many cases, it is difficult to construct Image Specification on the full dataset. 
By randomly sampling a subset of the dataset, we can construct Image Specification based on it efficiently, with a strong enough statistical description of the full dataset.

.. tip::
   Typically, sampling 3,000 to 10,000 images is sufficient to generate the Image Specification.

.. code-block:: python

   import torchvision
   from torch.utils.data import DataLoader
   from learnware.specification import generate_rkme_image_spec


   cifar10 = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()
   )
   X, _ = next(iter(DataLoader(cifar10, batch_size=len(cifar10))))

   spec = generate_rkme_image_spec(X, sample_size=5000)
   spec.save("cifar10.json")

Privacy Protection
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the third row of the figure, we show the eight pseudo-data with the largest weights :math:`\beta` in the Image Specification generated on the CIFAR-10 dataset.
Notice that the Image Specification generated based on Neural Tangent Kernel (NTK) protects the user's privacy very well.

In contrast, we show the performance of the RBF kernel on image dat in the first row of the figure below. 
The RBF not only exposes the real data (plotted in the corresponding position in the second row), but also fails to fully utilise the weights :math:`\beta`.

.. image:: ../_static/img/image_spec.png
   :align: center

Text Specification
--------------------------

Different from tabular data, each text input is a string of different length, so we should first transform them to equal-length arrays. Sentence embedding is used here to complete this transformation. We choose the model ``paraphrase-multilingual-MiniLM-L12-v2``, a lightweight multilingual embedding model. Then, we calculate the RKME specification on the embedding,  just like we do with tabular data. Besides, we use the package ``langdetect`` to detect and store the language of the text inputs for further search. We hope to search for the learnware which supports the language of the user task.

System Specification
======================================

In contrast to ``regular specification``\ s which are generated solely by users,
``system specification``\ s are higher-level statistical specifications assigned by learnware markets 
to effectively accommodate and organize heterogeneous learnwares. 
This implies that ``regular specification``\ s are usually applicable across different markets, while ``system specification``\ s are generally closely associated
with particular learnware market implementations.

``system specification`` play a critical role in heterogeneous markets such as the ``Hetero Market``:

- Learnware organizers use these specifications to connect isolated specification islands into unified "specification world"s.
- Learnware searchers perform helpful learnware recommendations among all table learnwares in the market, leveraging the ``system specification``\ s generated for users.


``learnware`` package now includes a type of ``system specification``, named ``HeteroMapTableSpecification``, made especially for the ``Hetero Market`` implementation.
This specification is automatically given to all table learnwares when they are added to the ``Hetero Market``.
It is also set up to be updated periodically, ensuring it remains accurate as the learnware market evolves and builds more precise specification worlds.
Please refer to `COMPONENTS: Hetero Market  <../components/market.html#hetero-market>`_ for implementation details.