.. _dev:
================
For Developer
================

Docstring
============
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

You can fix the bug by inputting the following code in the command line.


Continuous Integration
======================
Continuous Integration (CI) tools help you stick to the quality standards by running tests every time you push a new commit and reporting the results to a pull request.

``Learnware Market`` will check the following tests when you pull a request:
1. We will check your code style pylint, you can fix your code style by the following commands:

.. code-block:: bash

    pip install black
    python -m black . -l 120


2. We will check the pytest, you commit should can pass all tests in the tests directory. Run the following commands to check:

.. code-block:: bash

    pip install pytest
    python -m pytest tests

Development Guidance
=================================

As a developer, you often want make changes to ``Learnware Market`` and hope it would reflect directly in your environment without reinstalling it. You can install ``Learnware Market`` in editable mode with following command.

- For Windows and Linux users:

    .. code-block:: bash
        
        $ git clone https://git.nju.edu.cn/learnware/learnware-market.git && cd learnware-market
        $ python setup.py install

- For macOS users:

    .. code-block:: bash
        
        $ conda install -c pytorch faiss
        $ git clone https://git.nju.edu.cn/learnware/learnware-market.git && cd learnware-market
        $ python setup.py install