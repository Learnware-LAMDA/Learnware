from .base import LearnwareMarket
from .easy2 import EasyOrganizer, EasySearcher, EasySemanticChecker, EasyStatChecker
from .hetergeneous import HeteroMapTableOrganizer, HeteroMapTableSearcher

MARKET_CONFIG = {
    "easy": {
        "organizer": EasyOrganizer(),
        "searcher": EasySearcher(),
        "checker_list": [EasySemanticChecker(), EasyStatChecker()],
    }, 
    "hetero": {
        "organizer": HeteroMapTableOrganizer(),
        "searcher": HeteroMapTableSearcher(),
        "checker_list": []
    }
}


def instantiate_learnware_market(market_id="default", name="easy", **kwargs):
    return LearnwareMarket(
        market_id=market_id,
        organizer=MARKET_CONFIG[name]["organizer"],
        searcher=MARKET_CONFIG[name]["searcher"],
        checker_list=MARKET_CONFIG[name]["checker_list"],
        **kwargs
    )
