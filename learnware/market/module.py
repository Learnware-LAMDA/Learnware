from .base import LearnwareMarket
from .easy2 import EasyChecker, EasyOrganizer, EasySearcher

MARKET_CONFIG = {
    "easy": {
        "organizer": EasyOrganizer(),
        "checker": EasyChecker(),
        "searcher": EasySearcher(),
    }
}


def instatiate_learnware_market(market_id, name="easy", **kwargs):
    return LearnwareMarket(
        market_id=market_id,
        organizer=MARKET_CONFIG[name]["organizer"],
        checker=MARKET_CONFIG[name]["checker"],
        searcher=MARKET_CONFIG[name]["searcher"],
        **kwargs
    )
