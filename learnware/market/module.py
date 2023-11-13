from .base import LearnwareMarket
from .easy import EasyOrganizer, EasySearcher, EasySemanticChecker, EasyStatChecker
from .heterogeneous import HeteroMapTableOrganizer, HeteroSearcher


def get_market_config():
    market_config = {
        "easy": {
            "organizer": EasyOrganizer(),
            "searcher": EasySearcher(),
            "checker_list": [EasySemanticChecker(), EasyStatChecker()],
        },
        "hetero": {
            "organizer": HeteroMapTableOrganizer(),
            "searcher": HeteroSearcher(),
            "checker_list": [EasySemanticChecker(), EasyStatChecker()],
        },
    }
    return market_config


def instantiate_learnware_market(
    market_id="default",
    name="easy",
    rebuild=False,
    organizer_kwargs=None,
    searcher_kwargs=None,
    checker_kwargs=None,
    **kwargs
):
    market_config = get_market_config()
    return LearnwareMarket(
        market_id=market_id,
        organizer=market_config[name]["organizer"],
        searcher=market_config[name]["searcher"],
        checker_list=market_config[name]["checker_list"],
        organizer_kwargs=organizer_kwargs,
        searcher_kwargs=searcher_kwargs,
        checker_kwargs=checker_kwargs,
        **kwargs
    )
