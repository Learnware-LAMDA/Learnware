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
from typing import Union, List

from ..config import C
from .. import learnware
from .container import LearnwaresContainer
from ..market import BaseChecker, EasySemanticChecker, EasyStatChecker
from ..logger import get_module_logger
from ..specification import Specification
from ..learnware import get_learnware_from_dirpath
from ..tests import get_semantic_specification

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


class LearnwareClient:
    def __init__(self, host=None):
        self.headers = None

        if host is None:
            self.host = C.backend_host
        else:
            self.host = host

        self.chunk_size = 1024 * 1024
        self.tempdir_list = []
        atexit.register(self.cleanup)

    def login(self, email, token):
        url = f"{self.host}/auth/login_by_token"

        response = requests.post(url, json={"email": email, "token": token})

        result = response.json()
        if result["code"] != 0:
            raise Exception("login failed: " + json.dumps(result))

        token = result["data"]["token"]
        self.headers = {"Authorization": f"Bearer {token}"}

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
        assert self._check_semantic_specification(semantic_specification), "Semantic specification check failed!"
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

    def download_learnware(self, learnware_id, save_path):
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
    def search_learnware(self, specification: Specification, page_size=10, page_index=0):
        url = f"{self.host}/engine/search_learnware"

        stat_spec = specification.get_stat_spec()
        if len(stat_spec) > 1:
            raise Exception("statistical specification must have only one key.")

        if len(stat_spec) == 1:
            stat_spec = list(stat_spec.values())[0]
        else:
            stat_spec = None

        returns = []
        with tempfile.NamedTemporaryFile(prefix="learnware_stat_", suffix=".json") as ftemp:
            if stat_spec is not None:
                stat_spec.save(ftemp.name)

            with open(ftemp.name, "r") as fin:
                semantic_specification = specification.get_semantic_spec()
                if semantic_specification is None:
                    semantic_specification = {}

                if stat_spec is None:
                    files = None
                else:
                    files = {"statistical_specification": fin}

                response = requests.post(
                    url,
                    files=files,
                    data={
                        "semantic_specification": json.dumps(specification.get_semantic_spec()),
                        "limit": page_size,
                        "page": page_index,
                    },
                    headers=self.headers,
                )

                result = response.json()

                if result["code"] != 0:
                    raise Exception("search failed: " + json.dumps(result))

                for learnware in result["data"]["learnware_list_single"]:
                    returns.append(
                        {
                            "learnware_id": learnware["learnware_id"],
                            "semantic_specification": learnware["semantic_specification"],
                            "matching": learnware["matching"],
                        }
                    )

        return returns

    @require_login
    def delete_learnware(self, learnware_id):
        url = f"{self.host}/user/delete_learnware"
        response = requests.post(url, json={"learnware_id": learnware_id}, headers=self.headers)

        result = response.json()

        if result["code"] != 0:
            raise Exception("delete failed: " + json.dumps(result))

    def create_semantic_specification(
        self,
        name,
        description,
        data_type,
        task_type,
        library_type,
        scenarios,
        input_description=None,
        output_description=None,
    ):
        semantic_specification = dict()
        semantic_specification["Data"] = {"Type": "Class", "Values": [data_type]}
        semantic_specification["Task"] = {"Type": "Class", "Values": [task_type]}
        semantic_specification["Library"] = {"Type": "Class", "Values": [library_type]}
        semantic_specification["Scenario"] = {"Type": "Tag", "Values": scenarios}
        semantic_specification["Name"] = {"Type": "String", "Values": name}
        semantic_specification["Description"] = {"Type": "String", "Values": description}
        semantic_specification["Input"] = input_description
        semantic_specification["Output"] = output_description

        return semantic_specification

    def list_semantic_specification_values(self, key: SemanticSpecificationKey):
        url = f"{self.host}/engine/semantic_specification"
        response = requests.get(url, headers=self.headers)
        result = response.json()
        semantic_conf = result["data"]["semantic_specification"]
        return semantic_conf[key.value]["Values"]

    def load_learnware(
        self,
        learnware_path: Union[str, List[str]] = None,
        learnware_id: Union[str, List[str]] = None,
        runnable_option: str = None,
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
            raise logger.warning(f"runnable_option must be one of ['conda', 'docker'], but got {runnable_option}")

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

            return learnware.get_learnware_from_dirpath(learnware_id, semantic_specification, tempdir)

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

        if len(learnware_list) == 1:
            return learnware_list[0]
        else:
            return learnware_list

    @staticmethod
    def _check_semantic_specification(semantic_spec):
        return EasySemanticChecker.check_semantic_spec(semantic_spec)[0] != BaseChecker.INVALID_LEARNWARE

    @staticmethod
    def _check_stat_specification(learnware):
        from ..market import CondaChecker

        stat_checker = CondaChecker(inner_checker=EasyStatChecker())
        return stat_checker(learnware)[0] != BaseChecker.INVALID_LEARNWARE

    @staticmethod
    def check_learnware(learnware_zip_path, semantic_specification=None):
        semantic_specification = (
            get_semantic_specification() if semantic_specification is None else semantic_specification
        )

        assert LearnwareClient._check_semantic_specification(
            semantic_specification
        ), "Semantic specification check failed!"

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            with zipfile.ZipFile(learnware_zip_path, mode="r") as z_file:
                z_file.extractall(tempdir)

            learnware = get_learnware_from_dirpath(
                id="test", semantic_spec=semantic_specification, learnware_dirpath=tempdir
            )

            if learnware is None:
                raise Exception("The learnware is not valid.")

            assert LearnwareClient._check_stat_specification(learnware), "Stat specification check failed!"

        logger.info("The learnware has passed the test.")

    def cleanup(self):
        for tempdir in self.tempdir_list:
            tempdir.cleanup()
