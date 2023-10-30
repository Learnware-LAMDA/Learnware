from .base import LearnwareMarket
from .easy2 import EasyOrganizer, EasySearcher, EasySemanticChecker, EasyStatisticalChecker

MARKET_CONFIG = {
    "easy": {
        "organizer": EasyOrganizer(),
        "searcher": EasySearcher(),
        "checker_list": [EasySemanticChecker(), EasyStatisticalChecker()],
    }
}


def instatiate_learnware_market(market_id, name="easy", **kwargs):
    return LearnwareMarket(
        market_id=market_id,
        organizer=MARKET_CONFIG[name]["organizer"],
        searcher=MARKET_CONFIG[name]["searcher"],
        checker_list=MARKET_CONFIG[name]["checker_list"],
        **kwargs
    )