import os
import uuid
import yaml
import json
import atexit
import zipfile
import hashlib
import requests
import tempfile
from enum import Enum
from tqdm import tqdm
from typing import Union, List, Optional

from ..config import C
from .container import LearnwaresContainer
from ..market import BaseChecker
from ..specification import generate_semantic_spec
from ..logger import get_module_logger
from ..learnware import get_learnware_from_dirpath
from ..market import BaseUserInfo


CHUNK_SIZE = 1024 * 1024
logger = get_module_logger(module_name="LearnwareClient")


def require_login(func):
    def wrapper(self, *args, **kwargs):
        if self.headers is None:
            raise Exception("Please login first.")
        return func(self, *args, **kwargs)

    return wrapper


def file_chunks(file_path):
    with open(file_path, "rb") as fin:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def compute_file_hash(file_path):
    file_hash = hashlib.md5()
    for chunk in file_chunks(file_path):
        file_hash.update(chunk)
    return file_hash.hexdigest()


class SemanticSpecificationKey(Enum):
    DATA_TYPE = "Data"
    TASK_TYPE = "Task"
    LIBRARY_TYPE = "Library"
    SENARIOES = "Scenario"
    LICENSE = "License"


class LearnwareClient:
    def __init__(self, host=None):
        self.headers = None

        if host is None:
            self.host = C.backend_host
        else:
            self.host = host

        self.chunk_size = 1024 * 1024
        self.tempdir_list = []
        self.login_status = False
        atexit.register(self.cleanup)

    def is_connected(self):
        url = f"{self.host}/auth/login_by_token"
        response = requests.post(url)
        if response.status_code == 404:
            return False
        return True

    def login(self, email, token):
        url = f"{self.host}/auth/login_by_token"

        response = requests.post(url, json={"email": email, "token": token})

        result = response.json()
        if result["code"] != 0:
            raise Exception("login failed: " + json.dumps(result))

        token = result["data"]["token"]
        self.headers = {"Authorization": f"Bearer {token}"}
        self.login_status = True

    def is_login(self):
        return self.login_status

    @require_login
    def logout(self):
        url = f"{self.host}/auth/logout"
        response = requests.post(url, headers=self.headers)
        result = response.json()
        if result["code"] != 0:
            raise Exception("logout failed: " + json.dumps(result))
        self.headers = None

    @require_login
    def upload_learnware(self, learnware_zip_path, semantic_specification):
        assert self._check_semantic_specification(semantic_specification)[0], "Semantic specification check failed!"
        file_hash = compute_file_hash(learnware_zip_path)
        url_upload = f"{self.host}/user/chunked_upload"

        num_chunks = os.path.getsize(learnware_zip_path) // CHUNK_SIZE + 1
        bar = tqdm(total=num_chunks, desc="Uploading", unit="MB")
        begin = 0
        for chunk in file_chunks(learnware_zip_path):
            response = requests.post(
                url_upload,
                files={
                    "chunk_file": chunk,
                },
                data={
                    "file_hash": file_hash,
                    "chunk_begin": begin,
                },
                headers=self.headers,
            )

            result = response.json()

            if result["code"] != 0:
                raise Exception("upload failed: " + json.dumps(result))

            begin += len(chunk)
            bar.update(1)
        bar.close()

        url_add = f"{self.host}/user/add_learnware_uploaded"

        response = requests.post(
            url_add,
            json={
                "file_hash": file_hash,
                "semantic_specification": json.dumps(semantic_specification),
            },
            headers=self.headers,
        )

        result = response.json()

        if result["code"] != 0:
            raise Exception("upload failed: " + json.dumps(result))

        return result["data"]["learnware_id"]

    @require_login
    def update_learnware(self, learnware_id, semantic_specification, learnware_zip_path=None):
        assert self._check_semantic_specification(semantic_specification)[0], "Semantic specification check failed!"

        url_update = f"{self.host}/user/update_learnware"
        payload = {"learnware_id": learnware_id, "semantic_specification": json.dumps(semantic_specification)}

        if learnware_zip_path is None:
            response = requests.post(
                url_update,
                files={"learnware_file": None},
                data=payload,
                headers=self.headers,
            )
        else:
            response = requests.post(
                url_update,
                files={"learnware_file": open(learnware_zip_path, "rb")},
                data=payload,
                headers=self.headers,
            )

        result = response.json()

        if result["code"] != 0:
            raise Exception("update failed: " + json.dumps(result))

    def download_learnware(self, learnware_id: str, save_path: str):
        url = f"{self.host}/engine/download_learnware"

        response = requests.get(
            url,
            params={
                "learnware_id": learnware_id,
            },
            headers=self.headers,
            stream=True,
        )

        if response.status_code != 200:
            raise Exception("download failed: " + json.dumps(response.json()))

        num_chunks = int(response.headers["Content-Length"]) // CHUNK_SIZE + 1
        bar = tqdm(total=num_chunks, desc="Downloading", unit="MB")

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(1)

    @require_login
    def list_learnware(self):
        url = f"{self.host}/user/list_learnware"
        response = requests.post(url, json={"page": 0, "limit": 10000}, headers=self.headers)

        result = response.json()

        if result["code"] != 0:
            raise Exception("list failed: " + json.dumps(result))

        learnware_list = result["data"]["learnware_list"]

        return learnware_list

    @require_login
    def search_learnware(self, user_info: BaseUserInfo, page_size=10, page_index=0):
        url = f"{self.host}/engine/search_learnware"

        stat_spec = user_info.stat_info
        if len(stat_spec) > 1:
            raise Exception("statistical specification must have only one key.")

        if len(stat_spec) == 1:
            stat_spec = list(stat_spec.values())[0]
        else:
            stat_spec = None

        returns = {
            "single": {
                "learnware_ids": [],
                "semantic_specifications": [],
                "matching": [],
            },
            "multiple": {
                "learnware_ids": [],
                "semantic_specifications": [],
                "matching": None,
            },
        }
        with tempfile.NamedTemporaryFile(prefix="learnware_stat_", suffix=".json", delete=False) as ftemp:
            temp_file_name = ftemp.name
            if stat_spec is not None:
                stat_spec.save(temp_file_name)

        with open(temp_file_name, "r") as fin:
            semantic_specification = user_info.get_semantic_spec()
            if stat_spec is None:
                files = None
            else:
                files = {"statistical_specification": fin}

            response = requests.post(
                url,
                files=files,
                data={
                    "semantic_specification": json.dumps(semantic_specification),
                    "limit": page_size,
                    "page": page_index,
                },
                headers=self.headers,
            )
            result = response.json()

            if result["code"] != 0:
                raise Exception("search failed: " + json.dumps(result))

            for learnware in result["data"]["learnware_list_single"]:
                returns["single"]["learnware_ids"].append(learnware["learnware_id"])
                returns["single"]["semantic_specifications"].append(learnware["semantic_specification"])
                returns["single"]["matching"].append(learnware["matching"])

            if len(result["data"]["learnware_list_multi"]) > 0:
                multi_learnware = result["data"]["learnware_list_multi"][0]
                returns["multiple"]["learnware_ids"].append(multi_learnware["learnware_id"])
                returns["multiple"]["semantic_specifications"].append(multi_learnware["semantic_specification"])
                returns["multiple"]["matching"] = learnware["matching"]

        # Delete temp json file
        os.remove(temp_file_name)

        return returns

    @require_login
    def delete_learnware(self, learnware_id):
        url = f"{self.host}/user/delete_learnware"
        response = requests.post(url, json={"learnware_id": learnware_id}, headers=self.headers)

        result = response.json()

        if result["code"] != 0:
            raise Exception("delete failed: " + json.dumps(result))

    def list_semantic_specification_values(self, key: SemanticSpecificationKey):
        url = f"{self.host}/engine/semantic_specification"
        response = requests.get(url, headers=self.headers)
        result = response.json()
        semantic_conf = result["data"]["semantic_specification"]
        return semantic_conf[key.value]["Values"]

    def load_learnware(
        self,
        learnware_path: Optional[Union[str, List[str]]] = None,
        learnware_id: Optional[Union[str, List[str]]] = None,
        runnable_option: Optional[str] = None,
    ):
        """Load learnware by learnware zip file or learnware id (zip file has higher priority)

        Parameters
        ----------
        learnware_path : Union[str, List[str]]
            learnware zip path or learnware zip path list
        learnware_id : Union[str, List[str]]
            learnware id or learnware id list
        runnable_option : str
            the option for instantiating learnwares
            - None: instantiate learnware without installing environment
            - "conda": instantiate learnware with installing conda virtual environment
            - "docker": instantiate learnware with creating docker container

        Returns
        -------
        Learnware
            The contructed learnware object or object list
        """
        if runnable_option is not None and runnable_option not in ["conda", "docker"]:
            raise ValueError(f"runnable_option must be one of ['conda', 'docker'], but got {runnable_option}")

        if learnware_path is None and learnware_id is None:
            raise ValueError("Requires one of learnware_path or learnware_id")

        def _get_learnware_by_id(_learnware_id):
            self.tempdir_list.append(tempfile.TemporaryDirectory(prefix="learnware_"))
            tempdir = self.tempdir_list[-1].name
            zip_path = os.path.join(tempdir, f"{str(uuid.uuid4())}.zip")
            self.download_learnware(_learnware_id, zip_path)
            return _get_learnware_by_path(zip_path, tempdir=tempdir)

        def _get_learnware_by_path(_learnware_zippath, tempdir=None):
            if tempdir is None:
                self.tempdir_list.append(tempfile.TemporaryDirectory(prefix="learnware_"))
                tempdir = self.tempdir_list[-1].name

            with zipfile.ZipFile(_learnware_zippath, "r") as z_file:
                z_file.extractall(tempdir)

            yaml_file = C.learnware_folder_config["yaml_file"]
            with open(os.path.join(tempdir, yaml_file), "r") as fin:
                learnware_info = yaml.safe_load(fin)

            learnware_id = learnware_info.get("id")
            if learnware_id is None:
                learnware_id = "test_id"

            semantic_specification = learnware_info.get("semantic_specification")
            if semantic_specification is None:
                semantic_specification = {}
            else:
                semantic_file = semantic_specification.get("file_name")

                with open(os.path.join(tempdir, semantic_file), "r") as fin:
                    semantic_specification = json.load(fin)

            return get_learnware_from_dirpath(learnware_id, semantic_specification, tempdir)

        learnware_list = []
        if learnware_path is not None:
            zip_paths = [learnware_path] if isinstance(learnware_path, str) else learnware_path

            for zip_path in zip_paths:
                learnware_obj = _get_learnware_by_path(zip_path)
                learnware_list.append(learnware_obj)

        elif learnware_id is not None:
            if isinstance(learnware_id, str):
                id_list = [learnware_id]
            elif isinstance(learnware_id, list):
                id_list = learnware_id

            for idx in id_list:
                learnware_obj = _get_learnware_by_id(idx)
                learnware_list.append(learnware_obj)

        if runnable_option is not None:
            if runnable_option == "conda":
                with LearnwaresContainer(learnware_list, cleanup=False, mode="conda") as env_container:
                    learnware_list = env_container.get_learnwares_with_container()
            elif runnable_option == "docker":
                with LearnwaresContainer(learnware_list, cleanup=False, mode="docker") as env_container:
                    learnware_list = env_container.get_learnwares_with_container()

        single_flag = isinstance(learnware_path, str) if learnware_path is not None else isinstance(learnware_id, str)
        return learnware_list[0] if single_flag else learnware_list

    @staticmethod
    def _check_semantic_specification(semantic_spec):
        from ..market import EasySemanticChecker

        check_status, message = EasySemanticChecker.check_semantic_spec(semantic_spec)
        return check_status != BaseChecker.INVALID_LEARNWARE, message

    @staticmethod
    def _check_stat_specification(learnware):
        from ..market import EasyStatChecker, CondaChecker

        stat_checker = CondaChecker(inner_checker=EasyStatChecker())
        check_status, message = stat_checker(learnware)
        return check_status != BaseChecker.INVALID_LEARNWARE, message

    @staticmethod
    def check_learnware(learnware_zip_path, semantic_specification=None):
        semantic_specification = generate_semantic_spec(
            name="test",
            description="test",
            data_type="Text",
            task_type="Segmentation",
            scenarios="Financial",
            library_type="Scikit-learn",
            license="Apache-2.0",
        ) if semantic_specification is None else semantic_specification
        
        check_status, message = LearnwareClient._check_semantic_specification(semantic_specification)
        assert check_status, f"Semantic specification check failed due to {message}!"

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            with zipfile.ZipFile(learnware_zip_path, mode="r") as z_file:
                z_file.extractall(tempdir)

            learnware = get_learnware_from_dirpath(
                id="test", semantic_spec=semantic_specification, learnware_dirpath=tempdir, ignore_error=False
            )
            
            check_status, message = LearnwareClient._check_stat_specification(learnware)
            assert check_status is True, message

        logger.info("The learnware has passed the test.")

    def cleanup(self):
        for tempdir in self.tempdir_list:
            tempdir.cleanup()
