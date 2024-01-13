.. _beimingwu:
====================
Beimingwu System
====================

`Beimingwu System <https://bmwu.cloud/>`_ is based on the learnware paradigm, which systematically implements the entire process of learnware from submission to deployment, helping users effectively search and reuse learnwares without the need to build machine learning models from scratch.

The ``learnware`` package is the cornerstone of the Beimingwu system, functioning as its core engine.
It offers a comprehensive suite of central APIs that encompass a wide range of functionalities, including the submission, verification, organization, search, and deployment of learnware.
This integration ensures a streamlined and efficient process, facilitating seamless interactions within the system.

Core Features in the Beimingwu System
=======================================

Beimingwu systematically implements the core process of the learnware paradigm for the first time:

- ``Submitting Stage``: The system includes multiple detection mechanisms to ensure the quality of uploaded learnwares. Additionally, the system trains a heterogeneous engine based on existing learnware specifications in the system to merge different specification islands and assign new specifications to learnwares. With more learnwares are submitted, the heterogeneous engine will continue to update, achieving continuous iteration of learnware specifications and building a more precise specification world.
- ``Deploying Stage``: After users upload task requirements, the system automatically selects whether to recommend a single learnware or multiple learnware combinations and provides efficient deployment methods. Whether it's a single learnware or a combination of multiple learnwares, the system offers convenient learnware reuse tools.

In addition, the Beimingwu system also has the following features:

- ``Learnware Specification Generation``: The Beimingwu system provides specification generation interfaces in the learnware package, supporting various data types (tables, images, and text) for efficient local generation.
- ``Learnware Quality Inspection``: The Beimingwu system includes multiple detection mechanisms to ensure the quality of each learnware in the system.
- ``Diverse Learnware Search``: The Beimingwu system supports both semantic specifications and statistical specifications searches, covering data types such as tables, images, and text. In addition, for table-based tasks, the system also supports the search for heterogeneous table learnwares.
- ``Local Learnware Deployment``: The Beimingwu system provides interfaces for learnware deployment and learnware reuse in the learnware package, facilitating users' convenient and secure learnware deployment.
- ``Data Privacy Protection``: The Beimingwu system operations, including learnware upload, search, and deployment, do not require users to upload local data. All relevant statistical specifications are generated locally by users, ensuring data privacy.
- ``Open Source System``: The Beimingwu system's source code is open-source, including the learnware package and frontend/backend code. The learnware package is highly extensible, making it easy to integrate new specification designs, learnware system designs, and learnware reuse methods in the future.

Beimingwu is the first system-level implementation of the learnware paradigm.
This pioneering venture is just the beginning, with vast opportunities for enhancement and growth in the related technological fields still ahead.