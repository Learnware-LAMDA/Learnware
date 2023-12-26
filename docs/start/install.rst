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


Also, Users can install ``Learnware`` by the source code according to the following steps:

- Enter the root directory of ``Learnware``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``Learnware``:

    .. code-block:: bash
        
        $ git clone hhttps://github.com/Learnware-LAMDA/Learnware.git && cd learnware
        $ python setup.py install

.. note::
   It's recommended to use anaconda/miniconda to setup the environment.

Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import learnware
   >>> learnware.__version__
   <LATEST VERSION>