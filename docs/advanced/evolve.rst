==============================
Evolvable Specification
==============================

The specification is the core of the learnware paradigm.
Once the learnware market decides to accept a submitted model, it will assign to the model a specification, which conveys the specialty and utility of the model in some format, without leaking its original training data.
As the number of learnwares in the market increases, the knowledge held in the learnware market is being continually enriched.
This growth makes it possible for specification evolvement, enabling the market to generate new specifications for each learnware that more accurately characterize the properties of each model and its relationships with others.
As a result, the learnware market can more effectively identify learnwares beneficial for user tasks.

To achieve evolvable specifications, you need to implement the class ``EvolvedOrganizer`` in ``learnware/market/evolve/``:

- First, design a method for the learnware market to generate new statistical specifications for learnwares and implement the function ``EvolvedOrganizer.generate_new_stat_specification``.
- Second, use the function ``EvolvedOrganizer.generate_new_stat_specification`` to implement the function ``EvolvedOrganizer.evolve_learnware_list``, which enables learnwares to evolve by assigning new statistical specifications.

When implementing the anchor design, it is essential to develop an appropriate evolvement method for anchor learnwares based on the specific anchor selection method.
In the anchor design, the learnware market sends anchor learnware to users, who then provide statistical information about the anchor learnwares on their tasks to the market.
Based on this statistical feedback from users, the market can more accurately characterize anchor learnwares and continuously evolve them.

To realize evolvable specifications, including anchor learnwares, you need to additionally implement the class ``EvolvedAnchoredOrganizer`` in ``learnware/market/evolve_anchor/``:

- First, based on the specific anchor selection method, design an appropriate evolvement method for anchor learnwares and implement the function ``EvolvedAnchoredOrganizer.evolve_anchor_learnware_list``.
- Second, utilize the statistical feedback from users to implement the function ``EvolvedAnchoredOrganizer.evolve_anchor_learnware_by_user``, which enables anchor learnwares to evolve continually as users interact with the learnware market.