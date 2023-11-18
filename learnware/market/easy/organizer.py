import os
import copy
import zipfile
import tempfile
from shutil import copyfile, rmtree
from typing import Tuple, List, Union, Dict

from .database_ops import DatabaseOperations
from ..base import BaseOrganizer, BaseChecker
from ...config import C as conf
from ...logger import get_module_logger
from ...learnware import Learnware, get_learnware_from_dirpath
from ...logger import get_module_logger

logger = get_module_logger("easy_organizer")


class EasyOrganizer(BaseOrganizer):
    def reload_market(self, rebuild=False) -> bool:
        """Reload the learnware organizer when server restarted.

        Returns
        -------
        bool
            A flag indicating whether the market is reloaded successfully.
        """
        self.market_store_path = os.path.join(conf.market_root_path, self.market_id)
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.learnware_pool_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.learnware_pool_path, "unzipped_learnwares")
        self.learnware_list = {}  # id: Learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.use_flags = {}
        self.count = 0
        self.dbops = DatabaseOperations(conf.database_url, "market_" + self.market_id)

        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            try:
                self.dbops.clear_learnware_table()
                rmtree(self.learnware_pool_path)
            except Exception as err:
                logger.error(f"Clear current database failed due to {err}!!")

        os.makedirs(self.learnware_pool_path, exist_ok=True)
        os.makedirs(self.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(self.learnware_folder_pool_path, exist_ok=True)
        (
            self.learnware_list,
            self.learnware_zip_list,
            self.learnware_folder_list,
            self.use_flags,
            self.count,
        ) = self.dbops.load_market()

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None
    ) -> Tuple[str, int]:
        """Add a learnware into the market.

        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.
        check_status: int
            A flag indicating whether the learnware is usable.
        learnware_id: int
            A id in database for learnware
        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating the final learnware check_status
        """
        if check_status == BaseChecker.INVALID_LEARNWARE:
            logger.warning("Learnware is invalid!")
            return None, BaseChecker.INVALID_LEARNWARE

        semantic_spec = copy.deepcopy(semantic_spec)
        logger.info("Get new learnware from %s" % (zip_path))

        learnware_id = "%08d" % (self.count) if learnware_id is None else learnware_id
        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % (learnware_id))
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, learnware_id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))

        try:
            new_learnware = get_learnware_from_dirpath(
                id=learnware_id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            logger.warning("New learnware is not properly added!")
            try:
                os.remove(target_zip_dir)
                rmtree(target_folder_dir)
            except:
                pass
            return None, BaseChecker.INVALID_LEARNWARE

        if new_learnware is None:
            return None, BaseChecker.INVALID_LEARNWARE

        learnwere_status = check_status if check_status is not None else BaseChecker.NONUSABLE_LEARNWARE

        self.dbops.add_learnware(
            id=learnware_id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=learnwere_status,
        )

        self.learnware_list[learnware_id] = new_learnware
        self.learnware_zip_list[learnware_id] = target_zip_dir
        self.learnware_folder_list[learnware_id] = target_folder_dir
        self.use_flags[learnware_id] = learnwere_status
        self.count += 1
        return learnware_id, learnwere_status

    def delete_learnware(self, id: str) -> bool:
        """Delete Learnware from market

        Parameters
        ----------
        id : str
            Learnware to be deleted

        Returns
        -------
        bool
            True for successful operation.
            False for id not found.
        """
        if not id in self.learnware_list:
            logger.warning("Learnware id:'{}' NOT Found!".format(id))
            return False

        zip_dir = self.learnware_zip_list[id]
        if os.path.exists(zip_dir):
            os.remove(zip_dir)

        folder_dir = self.learnware_folder_list[id]
        rmtree(folder_dir, ignore_errors=True)
        self.learnware_list.pop(id)
        self.learnware_zip_list.pop(id)
        self.learnware_folder_list.pop(id)
        self.use_flags.pop(id)
        self.dbops.delete_learnware(id=id)

        return True

    def update_learnware(self, id: str, zip_path: str = None, semantic_spec: dict = None, check_status: int = None):
        """Update learnware with zip_path, semantic_specification and check_status

        Parameters
        ----------
        id : str
            Learnware id
        zip_path : str, optional
            Filepath for learnware model, a zipped file.
        semantic_spec : dict, optional
            semantic_spec for new learnware, in dictionary format.
        check_status : int, optional
            A flag indicating whether the learnware is usable.

        Returns
        -------
        int
            The final learnware check_status.
        """
        if check_status == BaseChecker.INVALID_LEARNWARE:
            logger.warning("Learnware is invalid!")
            return BaseChecker.INVALID_LEARNWARE

        if zip_path is None and semantic_spec is None and check_status is None:
            logger.warning(
                "At least one of 'zip_path', 'semantic_spec' and 'check_status' should not be None when update learnware"
            )
            return BaseChecker.INVALID_LEARNWARE

        # Update semantic_specification
        learnware_zippath = self.learnware_zip_list[id] if zip_path is None else zip_path
        semantic_spec = (
            self.learnware_list[id].get_specification().get_semantic_spec() if semantic_spec is None else semantic_spec
        )
        self.dbops.update_learnware_semantic_specification(id, semantic_spec)

        # Update zip path
        target_zip_dir = self.learnware_zip_list[id]
        target_folder_dir = self.learnware_folder_list[id]
        if zip_path is not None:
            with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                with zipfile.ZipFile(zip_path, "r") as z_file:
                    z_file.extractall(tempdir)

                try:
                    new_learnware = get_learnware_from_dirpath(
                        id=id, semantic_spec=semantic_spec, learnware_dirpath=tempdir
                    )
                except Exception:
                    return BaseChecker.INVALID_LEARNWARE

                if new_learnware is None:
                    return BaseChecker.INVALID_LEARNWARE

            if zip_path != target_zip_dir:
                copyfile(zip_path, target_zip_dir)
            with zipfile.ZipFile(target_zip_dir, "r") as z_file:
                z_file.extractall(target_folder_dir)

        # Update check_status
        self.use_flags[id] = self.use_flags[id] if check_status is None else check_status
        self.dbops.update_learnware_use_flag(id, self.use_flags[id])

        # Update learnware list
        self.learnware_list[id] = get_learnware_from_dirpath(
            id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
        )

        return self.use_flags[id]

    def get_learnware_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """Search learnware by id or list of ids.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of target learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return target learnware or list of target learnwares.
            None for Learnware NOT Found.
        """
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_list:
                    ret.append(self.learnware_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

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
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_zip_list:
                    ret.append(self.learnware_zip_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_zip_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

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
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_folder_list:
                    ret.append(self.learnware_folder_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_folder_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

    def get_learnware_ids(self, top: int = None, check_status: int = None) -> List[str]:
        """Get learnware ids

        Parameters
        ----------
        top : int, optional
            The first top learnware ids to return, by default None
        check_status : bool, optional
            - None: return all learnware ids
            - Others: return learnware ids with check_status

        Returns
        -------
        List[str]
            Learnware ids
        """
        if check_status is None:
            filtered_ids = list(self.use_flags.keys())
        elif check_status in [BaseChecker.NONUSABLE_LEARNWARE, BaseChecker.USABLE_LEARWARE]:
            filtered_ids = [key for key, value in self.use_flags.items() if value == check_status]
        else:
            logger.warning(
                f"check_status must be in [{BaseChecker.NONUSABLE_LEARNWARE}, {BaseChecker.USABLE_LEARWARE}]!"
            )
            return None

        if top is None:
            return filtered_ids
        else:
            return filtered_ids[:top]

    def get_learnwares(self, top: int = None, check_status: int = None) -> List[Learnware]:
        """Get learnware list

        Parameters
        ----------
        top : int, optional
            The first top learnwares to return, by default None
        check_status : bool, optional
            - None: return all learnwares
            - Others: return learnwares with check_status

        Returns
        -------
        List[Learnware]
            Learnware list
        """
        learnware_ids = self.get_learnware_ids(top, check_status)
        return [self.learnware_list[idx] for idx in learnware_ids]

    def reload_learnware(self, learnware_id: str):
        if learnware_id not in self.learnware_list:
            self.count += 1

        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % (learnware_id))
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, learnware_id)
        self.learnware_zip_list[learnware_id] = target_zip_dir
        self.learnware_folder_list[learnware_id] = target_folder_dir
        semantic_spec = self.dbops.get_learnware_semantic_specification(learnware_id)
        self.learnware_list[learnware_id] = get_learnware_from_dirpath(
            id=learnware_id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
        )
        self.use_flags[learnware_id] = self.dbops.get_learnware_use_flag(learnware_id)

    def get_learnware_info_from_storage(self, learnware_id: str) -> Dict:
        """return learnware zip path and semantic_specification from storage

        Parameters
        ----------
        learnware_id : str
            learnware id

        Returns
        -------
        Dict
            - semantic_spec: semantic_specification
            - zip_path: zip_path
            - folder_path: folder_path
            - use_flag: use_flag
        """
        return self.dbops.get_learnware_info(learnware_id)

    def __len__(self):
        return len(self.learnware_list)
