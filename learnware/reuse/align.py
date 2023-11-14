from ..learnware import Learnware


class AlignLearnware(Learnware):
    """The aligned learnware class, providing the interfaces to align learnware and make predictions"""

    def __init__(self, learnware: Learnware):
        """The initialization method for align learnware

        Parameters
        ----------
        learnware : Learnware
            The learnware list to reuse and make predictions
        """
        super(AlignLearnware, self).__init__(
            id=learnware.id,
            model=learnware.get_model(),
            specification=learnware.get_specification(),
            learnware_dirpath=learnware.get_dirpath(),
        )

    def align(self):
        """Align the learnware with specification or data"""

        raise NotImplementedError("The align method is not implemented!")
