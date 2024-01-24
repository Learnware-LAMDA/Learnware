.. _beimingwu:
====================
Beimingwu System
====================

`Beimingwu <https://bmwu.cloud/>`_ is the first systematic open-source implementation of learnware dock system, providing a preliminary research platform for learnware studies. Developers worldwide can submit their models freely to the learnware dock. They can generate specifications for the model with the help of Beimingwu without disclosing their raw data, and then the model and specification can be assembled into a learnware, which will be accommodated in the learnware dock. Future users can solve their tasks by submitting their requirements and reusing helpful learnwares returned by Beimingwu, while also not disclosing their own data. It is anticipated that after Beimingwu accumulates millions of learnwares, an "emergent" behavior may occur: machine learning tasks that have never been specifically tackled may be solved by assembling and reusing some existing learnwares.

The ``learnware`` package is the cornerstone of the Beimingwu system, functioning as its core engine.
It offers a comprehensive suite of central APIs that encompass a wide range of functionalities, including the submission, verification, organization, search, and deployment of learnware.
This integration ensures a streamlined and efficient process, facilitating seamless interactions within the system.

Core Features in the Beimingwu System
=======================================

The Beimingwu learnware dock system, serving as a preliminary research platform for learnware, systematically implements the core processes of the learnware paradigm for the first time:

- ``Submitting Stage``: The system includes multiple detection mechanisms to ensure the quality of uploaded learnwares. Additionally, the system trains a heterogeneous engine based on existing learnware specifications in the system to merge different specification islands and assign new specifications to learnwares. With more learnwares are submitted, the heterogeneous engine will continue to update, achieving continuous iteration of learnware specifications and building a more precise specification world.
- ``Deploying Stage``: After users upload task requirements, the system automatically selects whether to recommend a single learnware or multiple learnware combinations and provides efficient deployment methods. Whether it's a single learnware or a combination of multiple learnwares, the system offers convenient learnware reuse tools.

In addition, the Beimingwu system also has the following features:

- ``Learnware Specification Generation``: The Beimingwu system provides specification generation interfaces in the learnware package, supporting various data types (tables, images, and text) for efficient local generation.
- ``Learnware Quality Inspection``: The Beimingwu system includes multiple detection mechanisms to ensure the quality of each learnware in the system.
- ``Diverse Learnware Search``: The Beimingwu system supports both semantic specifications and statistical specifications searches, covering data types such as tables, images, and text. In addition, for table-based tasks, the system also supports the search for heterogeneous table learnwares.
- ``Local Learnware Deployment``: The Beimingwu system provides interfaces for learnware deployment and learnware reuse in the learnware package, facilitating users' convenient and secure learnware deployment.
- ``Data Privacy Protection``: The Beimingwu system operations, including learnware upload, search, and deployment, do not require users to upload local data. All relevant statistical specifications are generated locally by users, ensuring data privacy.
- ``Open Source System``: The Beimingwu system's source code is open-source, including the learnware package and frontend/backend code. The learnware package is highly extensible, making it easy to integrate new specification designs, learnware system designs, and learnware reuse methods in the future.

Building the learnware paradigm requires collective efforts from the community. As the first learnware dock system, Beimingwu is still in its early stages, with much room for improvement in related technologies. We sincerely invite the community to upload models, collaborate in system development, and engage in research and enhancements in learnware algorithms. Your valuable feedback is essential for the continuous improvement of the system.