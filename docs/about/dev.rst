.. _dev:
================
For Developer
================

Install with Dev Mode
=======================

As a developer, you often want make changes to ``Learnware Market`` and hope it would reflect directly in your environment without reinstalling it. You can install ``Learnware Market`` in editable mode with following command.

.. code-block:: bash
    
    $ git clone https://github.com/Learnware-LAMDA/Learnware.git && cd learnware
    $ python setup.py install

Commit Format
==============

Please submit in the following manner: Submit using the format ``prefix`` + ``space`` + ``suffix``.
There are four choices for the prefix, and they can be combined using commas:

- [ENH]: Represents enhancement, indicating the addition of new features.
- [DOC]: Indicates modifications to the documentation.
- [FIX]: Represents bug fixes and typo corrections.
- [MNT]: Indicates other minor modifications, such as version updates.
  
The suffix specifies the specific nature of the modification, with the initial letter capitalized.

Examples: The following are all valid:

- [DOC] Fix the document
- [FIX, ENH] Fix the bug and add some feature"


Docstring
============
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

You can fix the bug by inputting the following code in the command line.


Continuous Integration
======================
Continuous Integration (CI) tools help you stick to the quality standards by running tests every time you push a new commit and reporting the results to a pull request.

``Learnware Market`` will check the following tests when you pull a request:
1. We will check your code length, you can fix your code style by the following commands:

.. code-block:: bash

    pip install black
    python -m black . -l 120


2. We will check the pytest, you commit should can pass all tests in the tests directory. Run the following commands to check:

.. code-block:: bash

    pip install pytest
    python -m pytest tests

``pre-commit`` Config
========================

The ``Learnware`` Package support config ``pre-commit``. Run the following command to install ``pre-commit``:

.. code-block:: bash

    pip install pre-commit


Run the following command in the root directory of ``Learnware`` Project to enable ``pre-commit``:

.. code-block:: bash

    pre-commit install

``isort`` Config
===================

The codes in the ``Learnware`` Package will be processed by ``isort`` (``examples`` and ``tests`` are excluded). Run the following command to install ``isort``:

.. code-block:: bash

    pip install isort

Run the following command in the root directory of ``Learnware`` Project to run ``isort``:

.. code-block:: bash

    isort learnware --reverse-relative

