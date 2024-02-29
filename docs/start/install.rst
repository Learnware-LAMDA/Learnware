.. _install:
========================
Installation Guide
========================


``learnware`` Package Installation
===================================
.. note::

   The ``learnware`` package supports `Windows`, `Linux`. It's recommended to use ``learnware`` in `Linux`. This package supports Python3, which is up to Python3.11.

Users can easily install ``learnware`` by pip according to the following command:

.. code-block:: bash

    pip install learnware

In the ``learnware`` package, besides the base classes, many core functionalities such as "learnware specification generation" and "learnware deployment" rely on the ``torch`` library. Users have the option to manually install ``torch``, or they can directly use the following command to install the ``learnware`` package:

.. code-block:: bash

    pip install learnware[full]

.. note:: 
    However, it's crucial to note that due to the potential complexity of the user's local environment, installing ``learnware[full]`` does not guarantee that ``torch`` will successfully invoke ``CUDA`` in the user's local setting.


Install ``learnware`` Package From Source
==========================================

Also, Users can install ``learnware`` by the source code according to the following steps:

- Enter the root directory of this project, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``learnware``:

    .. code-block:: bash
        
        $ git clone https://github.com/Learnware-LAMDA/Learnware.git && cd Learnware
        $ pip install -e .[dev]

.. note::
   It's recommended to use anaconda/miniconda to setup the environment. Also you can run ``pip install -e .[full, dev]`` to install ``torch`` automatically as well.

Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import learnware
   >>> learnware.__version__
   <LATEST VERSION>