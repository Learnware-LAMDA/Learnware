from .base import LearnwareMarket
from .classes import CondaChecker
from .easy import (
    EasyOrganizer,
    EasyFuzzSemanticSearcher,
    EasyStatSearcher,
    SeqCombinedSearcher,
    EasySemanticChecker,
    EasyStatChecker,
)
from .heterogeneous import HeteroMapTableOrganizer, HeteroStatSearcher
from .llm import LLMEasyOrganizer, LLMStatSearcher


def get_market_component(
    name, market_id, rebuild, organizer_kwargs=None, searcher_kwargs=None, checker_kwargs=None, conda_checker=False
):
    organizer_kwargs = {} if organizer_kwargs is None else organizer_kwargs
    searcher_kwargs = {} if searcher_kwargs is None else searcher_kwargs
    checker_kwargs = {} if checker_kwargs is None else checker_kwargs

    if name == "easy":
        easy_organizer = EasyOrganizer(market_id=market_id, rebuild=rebuild)

        semantic_searcher_list = [EasyFuzzSemanticSearcher(easy_organizer)]
        stat_searcher_list = [EasyStatSearcher(easy_organizer)]
        easy_searcher = SeqCombinedSearcher(
            organizer=easy_organizer,
            semantic_searcher_list=semantic_searcher_list,
            stat_searcher_list=stat_searcher_list,
        )

        easy_checker_list = [
            EasySemanticChecker(),
            EasyStatChecker() if conda_checker is False else CondaChecker(EasyStatChecker()),
        ]

        market_component = {
            "organizer": easy_organizer,
            "searcher": easy_searcher,
            "checker_list": easy_checker_list,
        }

    elif name == "hetero":
        hetero_organizer = HeteroMapTableOrganizer(market_id=market_id, rebuild=rebuild, **organizer_kwargs)

        semantic_searcher_list = [EasyFuzzSemanticSearcher(hetero_organizer)]
        stat_searcher_list = [HeteroStatSearcher(hetero_organizer), EasyStatSearcher(hetero_organizer)]
        hetero_searcher = SeqCombinedSearcher(
            organizer=hetero_organizer,
            semantic_searcher_list=semantic_searcher_list,
            stat_searcher_list=stat_searcher_list,
        )

        hetero_checker_list = [
            EasySemanticChecker(),
            EasyStatChecker() if conda_checker is False else CondaChecker(EasyStatChecker()),
        ]

        market_component = {
            "organizer": hetero_organizer,
            "searcher": hetero_searcher,
            "checker_list": hetero_checker_list,
        }

    elif name == "llm":
        llm_organizer = LLMEasyOrganizer(market_id=market_id, rebuild=rebuild, **organizer_kwargs)

        semantic_searcher_list = [EasyFuzzSemanticSearcher(llm_organizer)]
        stat_searcher_list = [
            LLMStatSearcher(llm_organizer),
            HeteroStatSearcher(llm_organizer),
            EasyStatSearcher(llm_organizer),
        ]

        llm_searcher = SeqCombinedSearcher(
            organizer=llm_organizer,
            semantic_searcher_list=semantic_searcher_list,
            stat_searcher_list=stat_searcher_list,
        )

        llm_checker_list = [
            EasySemanticChecker(),
            EasyStatChecker() if conda_checker is False else CondaChecker(EasyStatChecker()),
        ]

        market_component = {
            "organizer": llm_organizer,
            "searcher": llm_searcher,
            "checker_list": llm_checker_list,
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
    conda_checker: bool = False,
    **kwargs,
):
    market_componets = get_market_component(
        name, market_id, rebuild, organizer_kwargs, searcher_kwargs, checker_kwargs, conda_checker
    )
    return LearnwareMarket(
        organizer=market_componets["organizer"],
        searcher=market_componets["searcher"],
        checker_list=market_componets["checker_list"],
        **kwargs,
    )
