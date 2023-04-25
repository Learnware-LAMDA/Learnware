========================
Installation Guide
========================


``Learnware Market`` Installation
=================================
.. note::

   ``Learnware Market`` supports `Windows`, `Linux` and `Macos`. It's recommended to use ``Learnware Market`` in `Linux`. ``Learnware Market`` supports Python3, which is up to Python3.8.

Users can easily install ``Learnware Market`` by pip according to the following command:

- For Windows and Linux users:

    .. code-block:: bash

        pip install learnware

- For macOS users:

    .. code-block:: bash

        conda install -c pytorch faiss
        pip install learnware


Also, Users can install ``Learnware Market`` by the source code according to the following steps:

- Enter the root directory of ``Learnware Market``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``Learnware Market``:

- For Windows and Linux users:

    .. code-block:: bash
        
        $ git clone https://git.nju.edu.cn/learnware/learnware-market.git && cd learnware-market
        $ python setup.py install

- For macOS users:

    .. code-block:: bash
        
        $ conda install -c pytorch faiss
        $ git clone https://git.nju.edu.cn/learnware/learnware-market.git && cd learnware-market
        $ python setup.py install

.. note::
   It's recommended to use anaconda/miniconda to setup the environment.

Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import learnware
   >>> learnware.__version__
   <LATEST VERSION>