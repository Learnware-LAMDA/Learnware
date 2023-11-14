from .base import LearnwareMarket
from .easy import EasyOrganizer, EasySearcher, EasySemanticChecker, EasyStatChecker
from .heterogeneous import HeteroMapTableOrganizer, HeteroSearcher


def get_market_component(name, market_id, rebuild, organizer_kwargs=None, searcher_kwargs=None, checker_kwargs=None):
    organizer_kwargs = {} if organizer_kwargs is None else organizer_kwargs
    searcher_kwargs = {} if searcher_kwargs is None else searcher_kwargs
    checker_kwargs = {} if checker_kwargs is None else checker_kwargs

    if name == "easy":
        easy_organizer = EasyOrganizer(market_id=market_id, rebuild=rebuild)
        easy_searcher = EasySearcher(organizer=easy_organizer)
        easy_checker_list = [EasySemanticChecker(), EasyStatChecker()]
        market_component = {
            "organizer": easy_organizer,
            "searcher": easy_searcher,
            "checker_list": easy_checker_list,
        }
    elif name == "hetero":
        hetero_organizer = HeteroMapTableOrganizer(market_id=market_id, rebuild=rebuild, **organizer_kwargs)
        hetero_searcher = HeteroSearcher(organizer=hetero_organizer)
        hetero_checker_list = [EasySemanticChecker(), EasyStatChecker()]

        market_component = {
            "organizer": hetero_organizer,
            "searcher": hetero_searcher,
            "checker_list": hetero_checker_list,
        }
    else:
        raise ValueError(f"name {name} is not supported for market")

    return market_component


def instantiate_learnware_market(
    market_id="default",
    name="easy",
    rebuild=False,
    organizer_kwargs: dict = None,
    searcher_kwargs: dict = None,
    checker_kwargs: dict = None,
    **kwargs,
):
    market_componets = get_market_component(name, market_id, rebuild, organizer_kwargs, searcher_kwargs, checker_kwargs)
    return LearnwareMarket(
        organizer=market_componets["organizer"],
        searcher=market_componets["searcher"],
        checker_list=market_componets["checker_list"],
        **kwargs,
    )
