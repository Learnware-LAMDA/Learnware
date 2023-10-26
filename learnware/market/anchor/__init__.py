

class AnchoredUserInfo(BaseUserInfo):
    """
    User Information for searching learnware (add the anchor design)

    - UserInfo contains the anchor list acquired from the market
    - UserInfo can update stat_info based on anchors
    """

    def __init__(self, id: str, semantic_spec: dict = dict(), stat_info: dict = dict()):
        super(AnchoredUserInfo, self).__init__(id, semantic_spec, stat_info)
        self.anchor_learnware_list = {}  # id: Learnware

    def add_anchor_learnware(self, learnware_id: str, learnware: Learnware):
        """Add the anchor learnware acquired from the market

        Parameters
        ----------
        learnware_id : str
            Id of anchor learnware
        learnware : Learnware
            Anchor learnware for capturing user requirements
        """
        self.anchor_learnware_list[learnware_id] = learnware

    def update_stat_info(self, name: str, item: Any):
        """Update stat_info based on anchor learnwares

        Parameters
        ----------
        name : str
            Name of stat_info
        item : Any
            Statistical information calculated on anchor learnwares
        """
        self.stat_info[name] = item
