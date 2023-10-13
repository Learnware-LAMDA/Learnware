import os
import numpy as np
import yaml
import json
import atexit
import zipfile
import hashlib
import requests
import tempfile
from enum import Enum
from tqdm import tqdm
from typing import List

from ..config import C
from .. import learnware
from . import package_utils
from .container import LearnwaresContainer
from ..market.easy import EasyMarket
from ..logger import get_module_logger
from ..specification import Specification
from ..learnware import BaseReuser, Learnware

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
            pass
        pass
    pass


def compute_file_hash(file_path):
    file_hash = hashlib.md5()
    for chunk in file_chunks(file_path):
        file_hash.update(chunk)
        pass
    return file_hash.hexdigest()


class SemanticSpecificationKey(Enum):
    DATA_TYPE = "Data"
    TASK_TYPE = "Task"
    LIBRARY_TYPE = "Library"
    SENARIOES = "Scenario"
    pass


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
        pass

    @require_login
    def logout(self):
        url = f"{self.host}/auth/logout"
        response = requests.post(url, headers=self.headers)
        result = response.json()
        if result["code"] != 0:
            raise Exception("logout failed: " + json.dumps(result))
        self.headers = None
        pass

    @require_login
    def upload_learnware(self, semantic_specification, learnware_file):
        file_hash = compute_file_hash(learnware_file)

        url_upload = f"{self.host}/user/chunked_upload"

        num_chunks = os.path.getsize(learnware_file) // CHUNK_SIZE + 1
        bar = tqdm(total=num_chunks, desc="Uploading", unit="MB")
        begin = 0
        for chunk in file_chunks(learnware_file):
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
            pass
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
                pass
            pass
        pass

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
            pass

        returns = []
        with tempfile.NamedTemporaryFile(prefix="learnware_stat_", suffix=".json") as ftemp:
            if stat_spec is not None:
                stat_spec.save(ftemp.name)
                pass

            with open(ftemp.name, "r") as fin:
                semantic_specification = specification.get_semantic_spec()
                if semantic_specification is None:
                    semantic_specification = {}
                    pass

                semantic_specification.pop("Input", None)
                semantic_specification.pop("Output", None)

                if stat_spec is None:
                    files = None
                else:
                    files = {"statistical_specification": fin}
                    pass

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
                    pass
                pass
            pass

        return returns

    @require_login
    def delete_learnware(self, learnware_id):
        url = f"{self.host}/user/delete_learnware"
        response = requests.post(url, json={"learnware_id": learnware_id}, headers=self.headers)

        result = response.json()

        if result["code"] != 0:
            raise Exception("delete failed: " + json.dumps(result))
        pass

    def check_learnware(self, path, semantic_specification):
        if os.path.isfile(path):
            with tempfile.TemporaryDirectory() as tempdir:
                with zipfile.ZipFile(path, "r") as z_file:
                    z_file.extractall(tempdir)
                    pass
                return self.check_learnware_folder(tempdir, semantic_specification)
            pass
        else:
            return self.check_learnware_folder(path, semantic_specification)
            pass
        pass

    def check_learnware_folder(self, folder, semantic_specification):
        learnware_obj = learnware.get_learnware_from_dirpath("test_id", semantic_specification, folder)

        check_result = EasyMarket.check_learnware(learnware_obj)
        if check_result == EasyMarket.USABLE_LEARWARE:
            return True
        else:
            return False
        pass

    def create_semantic_specification(
        self, name, description, data_type, task_type, library_type, senarioes, input_description, output_description
    ):
        semantic_specification = dict()
        semantic_specification["Input"] = input_description
        semantic_specification["Output"] = output_description
        semantic_specification["Data"] = {"Type": "Class", "Values": [data_type]}
        semantic_specification["Task"] = {"Type": "Class", "Values": [task_type]}
        semantic_specification["Library"] = {"Type": "Class", "Values": [library_type]}
        semantic_specification["Scenario"] = {"Type": "Tag", "Values": senarioes}
        semantic_specification["Name"] = {"Type": "String", "Values": name}
        semantic_specification["Description"] = {"Type": "String", "Values": description}

        return semantic_specification

    def list_semantic_specification_values(self, key: SemanticSpecificationKey):
        url = f"{self.host}/engine/semantic_specification"
        response = requests.get(url, headers=self.headers)
        result = response.json()
        semantic_conf = result["data"]["semantic_specification"]

        return semantic_conf[key.value]["Values"]

    def load_learnware(self, learnware_file: str, load_model: bool = True):
        self.tempdir_list.append(tempfile.TemporaryDirectory(prefix="learnware_"))
        tempdir = self.tempdir_list[-1].name

        with zipfile.ZipFile(learnware_file, "r") as z_file:
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

        learnware_obj = learnware.get_learnware_from_dirpath(learnware_id, semantic_specification, tempdir)

        if load_model:
            learnware_obj.instantiate_model()

        return learnware_obj

    def system(self, command):
        retcd = os.system(command)
        if retcd != 0:
            raise RuntimeError(f"Command {command} failed with return code {retcd}")
        pass

    def install_environment(self, zip_path, conda_env=None):
        """Install environment of a learnware

        Parameters
        ----------
        zip_path : str
            Path of the learnware zip file
        conda_env : optional
            If it is not None, a new conda environment will be created with the given name;
            If it is None, use current environment.

        Raises
        ------
        Exception
            Lack of the environment configuration file.
        """
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            with zipfile.ZipFile(zip_path, "r") as z_file:
                logger.info(f"zip_file namelist: {z_file.namelist}")
                if "environment.yaml" in z_file.namelist():
                    z_file.extract("environment.yaml", tempdir)
                    yaml_path = os.path.join(tempdir, "environment.yaml")
                    yaml_path_filter = os.path.join(tempdir, "environment_filter.yaml")
                    package_utils.filter_nonexist_conda_packages_file(yaml_path, yaml_path_filter)
                    # create environment
                    if conda_env is not None:
                        self.system(f"conda env update --name {conda_env} --file {yaml_path_filter}")
                        pass
                    else:
                        self.system(f"conda env update --file {yaml_path_filter}")
                        pass
                    pass
                elif "requirements.txt" in z_file.namelist():
                    z_file.extract("requirements.txt", tempdir)
                    requirements_path = os.path.join(tempdir, "requirements.txt")
                    requirements_path_filter = os.path.join(tempdir, "requirements_filter.txt")
                    package_utils.filter_nonexist_pip_packages_file(requirements_path, requirements_path_filter)

                    if conda_env is not None:
                        self.system(f"conda create -y --name {conda_env} python=3.8")
                        self.system(
                            f"conda run --name {conda_env} --no-capture-output python3 -m pip install -r {requirements_path_filter}"
                        )
                    else:
                        self.system(f"python3 -m pip install -r {requirements_path_filter}")
                        pass
                    pass
                else:
                    raise Exception("Environment.yaml or requirements.txt not found in the learnware zip file.")
                pass
            pass
        pass

    def test_learnware(self, zip_path, semantic_specification=None):
        if semantic_specification is None:
            semantic_specification = dict()
            pass

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            with zipfile.ZipFile(zip_path, mode="r") as z_file:
                z_file.extractall(tempdir)
                pass

            learnware_obj = learnware.get_learnware_from_dirpath("test_id", semantic_specification, tempdir)

            if learnware_obj is None:
                raise Exception("The learnware is not valid.")

            learnware_obj.instantiate_model()

            if len(semantic_specification) > 0:
                if EasyMarket.check_learnware(learnware_obj) != EasyMarket.USABLE_LEARWARE:
                    raise Exception("The learnware is not usable.")
                pass
            pass

        logger.info("test ok")
        pass

    def reuse_learnware(
        self,
        input_array: np.ndarray,
        learnware_list: List[Learnware],
        learnware_zippaths: List[str],
        reuser: BaseReuser,
    ):
        logger.info(f"reuse learnare list {learnware_list} with reuser {reuser}")
        with LearnwaresContainer(learnware_list, learnware_zippaths) as env_container:
            learnware_list = env_container.get_learnware_list_with_container()
            reuser.reset(learnware_list=learnware_list)
            result = reuser.predict(input_array)

        return result

    def cleanup(self):
        for tempdir in self.tempdir_list:
            tempdir.cleanup()
