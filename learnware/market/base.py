from __future__ import annotations

import traceback
import zipfile
import tempfile
from typing import Tuple, Any, List, Union
from ..learnware import Learnware, get_learnware_from_dirpath
from ..logger import get_module_logger

logger = get_module_logger("market_base", "INFO")


class BaseUserInfo:
    """User Information for searching learnware"""

    def __init__(self, id: str = None, semantic_spec: dict = None, stat_info: dict = None):
        """Initializing user information

        Parameters
        ----------
        id : str, optional
            user id, could be ignored in easy market
        semantic_spec : dict, optional
            semantic_spec selected by user, by default dict()
        stat_info : dict, optional
            statistical information uploaded by user, by default dict()
        """
        self.id = id
        self.semantic_spec = {} if semantic_spec is None else semantic_spec
        self.stat_info = {} if stat_info is None else stat_info

    def get_semantic_spec(self) -> dict:
        """Return user semantic specifications

        Returns
        -------
        dict
            user semantic specifications
        """
        return self.semantic_spec

    def get_stat_info(self, name: str):
        return self.stat_info.get(name, None)

    def update_stat_info(self, name: str, item: Any):
        """Update stat_info by market

        Parameters
        ----------
        name : str
            Name of stat_info
        item : Any
            Statistical information calculated by market
        """
        self.stat_info[name] = item


class LearnwareMarket:
    """Base interface for market, it provide the interface of search/add/detele/update learnwares"""

    def __init__(
        self,
        organizer: BaseOrganizer,
        searcher: BaseSearcher,
        checker_list: List[BaseChecker] = None,
        **kwargs,
    ):
        self.learnware_organizer = organizer
        self.learnware_searcher = searcher
        checker_list = [] if checker_list is None else checker_list
        self.learnware_checker = {checker.__class__.__name__: checker for checker in checker_list}

        for checker in self.learnware_checker.values():
            checker.reset(organizer=self.learnware_organizer)

    @property
    def market_id(self):
        return self.learnware_organizer.market_id

    def reset(self, organizer_kwargs=None, searcher_kwargs=None, checker_kwargs=None, **kwargs):
        if organizer_kwargs is not None:
            self.learnware_organizer.reset(**organizer_kwargs)

        if searcher_kwargs is not None:
            self.learnware_searcher.reset(**searcher_kwargs)

        if checker_kwargs is not None:
            if len(set(checker_kwargs) & set(self.learnware_checker)):
                for name, checker in self.learnware_checker.items():
                    checker.reset(**checker_kwargs.get(name, {}))
            else:
                for checker in self.learnware_checker.values():
                    checker.reset(**checker_kwargs)

        for _k, _v in kwargs.items():
            setattr(self, _k, _v)

    def reload_market(self, **kwargs) -> bool:
        self.learnware_organizer.reload_market(**kwargs)

    def check_learnware(self, zip_path: str, semantic_spec: dict, checker_names: List[str] = None, **kwargs) -> bool:
        try:
            final_status = BaseChecker.NONUSABLE_LEARNWARE
            if len(checker_names):
                with tempfile.TemporaryDirectory(prefix="pending_learnware_") as tempdir:
                    with zipfile.ZipFile(zip_path, mode="r") as z_file:
                        z_file.extractall(tempdir)

                    pending_learnware = get_learnware_from_dirpath(
                        id="pending", semantic_spec=semantic_spec, learnware_dirpath=tempdir
                    )
                    for name in checker_names:
                        checker = self.learnware_checker[name]
                        check_status, message = checker(pending_learnware)
                        final_status = max(final_status, check_status)

                        if check_status == BaseChecker.INVALID_LEARNWARE:
                            return BaseChecker.INVALID_LEARNWARE
            return final_status
        except Exception as err:
            traceback.print_exc()
            logger.warning(f"Check learnware failed! Due to {err}.")
            return BaseChecker.INVALID_LEARNWARE

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, checker_names: List[str] = None, **kwargs
    ) -> Tuple[str, int]:
        """Add a learnware into the market.

        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.
        checker_names : List[str], optional
            List contains checker names, by default None

        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating the final learnware check_status
        """
        checker_names = list(self.learnware_checker.keys()) if checker_names is None else checker_names
        check_status = self.check_learnware(zip_path, semantic_spec, checker_names)
        return self.learnware_organizer.add_learnware(
            zip_path=zip_path, semantic_spec=semantic_spec, check_status=check_status, **kwargs
        )

    def search_learnware(
        self, user_info: BaseUserInfo, check_status: int = None, **kwargs
    ) -> Tuple[Any, List[Learnware]]:
        """Search learnwares based on user_info from learnwares with check_status

        Parameters
        ----------
        user_info : BaseUserInfo
            User information for searching learnwares
        check_status : int, optional
            - None: search from all learnwares
            - Others: search from learnwares with check_status

        Returns
        -------
        Tuple[Any, List[Learnware]]
            Search results
        """
        return self.learnware_searcher(user_info, check_status, **kwargs)

    def delete_learnware(self, id: str, **kwargs) -> bool:
        return self.learnware_organizer.delete_learnware(id, **kwargs)

    def update_learnware(
        self,
        id: str,
        zip_path: str = None,
        semantic_spec: dict = None,
        checker_names: List[str] = None,
        check_status: int = None,
        **kwargs,
    ) -> int:
        """Update learnware with zip_path and semantic_specification

        Parameters
        ----------
        id : str
            Learnware id
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.
        checker_names : List[str], optional
            List contains checker names, by default None.
        check_status : int, optional
            A flag indicating whether the learnware is usable, by default None.

        Returns
        -------
        int
            The final learnware check_status.
        """
        zip_path = self.get_learnware_zip_path_by_ids(id) if zip_path is None else zip_path
        semantic_spec = (
            self.get_learnware_by_ids(id).get_specification().get_semantic_spec()
            if semantic_spec is None
            else semantic_spec
        )
        checker_names = list(self.learnware_checker.keys()) if checker_names is None else checker_names
        update_status = self.check_learnware(zip_path, semantic_spec, checker_names)
        check_status = (
            update_status if check_status is None or update_status == BaseChecker.INVALID_LEARNWARE else check_status
        )

        return self.learnware_organizer.update_learnware(
            id, zip_path=zip_path, semantic_spec=semantic_spec, check_status=check_status, **kwargs
        )

    def get_learnware_ids(self, top: int = None, check_status: int = None, **kwargs) -> List[str]:
        """get the list of learnware ids

        Parameters
        ----------
        top : int, optional
            The first top element to return, by default None
        check_status : int, optional
            - None: return all learnware ids
            - Others: return learnware ids with check_status

        Raises
        ------
        List[str]
            the first top ids
        """
        return self.learnware_organizer.get_learnware_ids(top, check_status, **kwargs)

    def get_learnwares(self, top: int = None, check_status: int = None, **kwargs) -> List[Learnware]:
        """get the list of learnwares

        Parameters
        ----------
        top : int, optional
            The first top element to return, by default None
        check_status : int, optional
            - None: return all learnwares
            - Others: return learnwares with check_status

        Raises
        ------
        List[Learnware]
            the first top learnwares
        """
        return self.learnware_organizer.get_learnwares(top, check_status, **kwargs)

    def reload_learnware(self, learnware_id: str):
        self.learnware_organizer.reload_learnware(learnware_id)

    def get_learnware_zip_path_by_ids(self, ids: Union[str, List[str]], **kwargs) -> Union[Learnware, List[Learnware]]:
        return self.learnware_organizer.get_learnware_zip_path_by_ids(ids, **kwargs)

    def get_learnware_dir_path_by_ids(self, ids: Union[str, List[str]], **kwargs) -> Union[Learnware, List[Learnware]]:
        return self.learnware_organizer.get_learnware_dir_path_by_ids(ids, **kwargs)

    def get_learnware_by_ids(self, id: Union[str, List[str]], **kwargs) -> Union[Learnware, List[Learnware]]:
        return self.learnware_organizer.get_learnware_by_ids(id, **kwargs)

    def __len__(self):
        return len(self.learnware_organizer)


class BaseOrganizer:
    def __init__(self, market_id, **kwargs):
        self.reset(market_id=market_id, **kwargs)

    def reset(self, market_id, rebuild=False, **kwargs):
        self.market_id = market_id
        self.reload_market(rebuild=rebuild, **kwargs)

    def reload_market(self, rebuild=False, **kwargs) -> bool:
        """Reload the learnware organizer when server restared.

        Returns
        -------
        bool
            A flag indicating whether the market is reload successfully.
        """

        raise NotImplementedError("reload market is Not Implemented in BaseOrganizer")

    def add_learnware(self, zip_path: str, semantic_spec: dict, check_status: int) -> Tuple[str, bool]:
        """Add a learnware into the market.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.


        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.

        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating what the flag of learnware is added.


        Raises
        ------
        FileNotFoundError
            file for model or statistical specification not found

        """
        raise NotImplementedError("add learnware is Not Implemented in BaseOrganizer")

    def delete_learnware(self, id: str) -> bool:
        """Delete a learnware from market

        Parameters
        ----------
        id : str
            id of learnware to be deleted

        Returns
        -------
        bool
            True if the target learnware is deleted successfully.

        Raises
        ------
        Exception
            Raise an excpetion when given id is NOT found in learnware list
        """
        raise NotImplementedError("delete learnware is Not Implemented in BaseOrganizer")

    def update_learnware(self, id: str, zip_path: str, semantic_spec: dict, check_status: int) -> bool:
        """
            Update Learnware with id and content to be updated.

        Parameters
        ----------
        id : str
            id of target learnware.
        """
        raise NotImplementedError("update learnware is Not Implemented in BaseOrganizer")

    def get_learnware_by_ids(self, id: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """
            Get Learnware from market by id

        Parameters
        ----------
        id : Union[str, List[str]]
            Given one id or a list of ids as target.

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return a Learnware object or a list of Learnware objects based on the type of input param.

            - The returned items are search results.
            - 'None' indicating the target id not found.
        """
        raise NotImplementedError("get_learnware_by_ids is not implemented in BaseOrganizer")

    def get_learnware_zip_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """Get Zipped Learnware file by id

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of targer learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return the path for target learnware or list of path.
            None for Learnware NOT Found.
        """
        raise NotImplementedError("get_learnware_zip_path_by_ids is not implemented in BaseOrganizer")

    def get_learnware_dir_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """Get Learnware dir path by id

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of targer learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return the dir path for target learnware or list of path.
            None for Learnware NOT Found.
        """
        raise NotImplementedError("get_learnware_dir_path_by_ids is not implemented in BaseOrganizer")

    def get_learnware_ids(self, top: int = None, check_status: int = None) -> List[str]:
        """get the list of learnware ids

        Parameters
        ----------
        top : int, optional
            The first top element to return, by default None
        check_status : int, optional
            - None: return all learnware ids
            - Others: return learnware ids with check_status

        Raises
        ------
        List[str]
            the first top ids
        """
        raise NotImplementedError("get_learnware_ids is not implemented in BaseOrganizer")

    def get_learnwares(self, top: int = None, check_status: int = None) -> List[Learnware]:
        """get the list of learnwares

        Parameters
        ----------
        top : int, optional
            The first top element to return, by default None
        check_status : int, optional
            - None: return all learnwares
            - Others: return learnwares with check_status

        Raises
        ------
        List[Learnware]
            the first top learnwares
        """
        raise NotImplementedError("get_learnwares is not implemented in BaseOrganizer")

    def __len__(self):
        raise NotImplementedError("__len__ is not implemented in BaseOrganizer")


class BaseSearcher:
    def __init__(self, organizer: BaseOrganizer, **kwargs):
        self.reset(organizer=organizer, **kwargs)

    def reset(self, organizer: BaseOrganizer, **kwargs):
        self.learnware_organizer = organizer

    def __call__(self, user_info: BaseUserInfo, check_status: int = None):
        """Search learnwares based on user_info from learnwares with check_status

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        check_status : int, optional
            - None: search from all learnwares
            - Others: search from learnwares with check_status
        """
        raise NotImplementedError("'__call__' method is not implemented in BaseSearcher")


class BaseChecker:
    INVALID_LEARNWARE = -1
    NONUSABLE_LEARNWARE = 0
    USABLE_LEARWARE = 1

    def reset(self, **kwargs):
        pass

    def __call__(self, learnware: Learnware) -> Tuple[int, str]:
        """Check the utility of a learnware

        Parameters
        ----------
        learnware : Learnware

        Returns
        -------
        Tuple[int, str]:
            flag and message of learnware check result
            - int
                A flag indicating whether the learnware can be accepted.
                - The INVALID_LEARNWARE denotes the learnware does not pass the check
                - The NOPREDICTION_LEARNWARE denotes the learnware pass the check but cannot make prediction due to some env dependency
                - The NOPREDICTION_LEARNWARE denotes the leanrware pass the check and can make prediction
            - str
                A message indicating the reason of learnware check result
        """

        raise NotImplementedError("'__call__' method is not implemented in BaseChecker")


class OrganizerRelatedChecker(BaseChecker):
    """Here this is the interface for checker who is related to the organizer"""

    def __init__(self, organizer: BaseOrganizer, **kwargs):
        self.reset(organizer=organizer, **kwargs)

    def reset(self, organizer: BaseOrganizer, **kwargs):
        self.learnware_organizer = organizer
