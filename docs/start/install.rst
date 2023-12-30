.. _install:
========================
Installation Guide
========================


``Learnware`` Package Installation
===================================
.. note::

   ``Learnware`` package supports `Windows`, `Linux`. It's recommended to use ``Learnware`` in `Linux`. ``Learnware`` supports Python3, which is up to Python3.11.

Users can easily install ``Learnware`` by pip according to the following command:

.. code-block:: bash

    pip install learnware

In the ``Learnware`` package, besides the base classes, many core functionalities such as "learnware specification generation" and "learnware deployment" rely on the ``torch`` library. Users have the option to manually install ``torch``, or they can directly use the following command to install the ``learnware`` package:

.. code-block:: bash

    pip install learnware[full]

.. note:: 
    However, it's crucial to note that due to the potential complexity of the user's local environment, installing ``learnware[full]`` does not guarantee that ``torch`` will successfully invoke ``CUDA`` in the user's local setting.


Install ``Learnware`` Package From Source
==========================================

Also, Users can install ``Learnware`` by the source code according to the following steps:

- Enter the root directory of ``Learnware``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``Learnware``:

    .. code-block:: bash
        
        $ git clone hhttps://github.com/Learnware-LAMDA/Learnware.git && cd Learnware
        $ pip install -e .[dev]

.. note::
   It's recommended to use anaconda/miniconda to setup the environment. Also you can run ``pip install -e .[full, dev]`` to install ``torch`` automatically as well.

Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import learnware
   >>> learnware.__version__
   <LATEST VERSION>