from ..specification import Specification
from ..config import C
import requests
import json
from tqdm import tqdm
import hashlib
import os
import tempfile


CHUNK_SIZE = 1024 * 1024


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


class LearnwareClient:
    def __init__(self, host=None):
        self.headers = None

        if host is None:
            self.host = C.backend_host
        else:
            self.host = host
            pass

        self.chunk_size = 1024 * 1024
        pass

    def login(self, email, password, hash_password=True):
        url = f"{self.host}/auth/login"

        if hash_password:
            password = hashlib.md5(password.encode()).hexdigest()
            pass
        
        response = requests.post(url, json={'email': email, 'password': password})

        result = response.json()
        if result['code'] != 0:
            raise Exception('login failed: ' + json.dumps(result))
        
    
        token = result['data']['token']
        self.headers = {'Authorization': f'Bearer {token}'}
        pass

    @require_login
    def logout(self):
        url = f"{self.host}/auth/logout"
        response = requests.post(url, headers=self.headers)
        result = response.json()
        if result['code'] != 0:
            raise Exception('logout failed: ' + json.dumps(result))
        self.headers = None
        pass
    
    @require_login
    def upload_learnware(self, semantic_specification, learnware_file):
        file_hash = compute_file_hash(learnware_file)

        url_upload = f"{self.host}/storage/chunked_upload"

        num_chunks = os.path.getsize(learnware_file) // CHUNK_SIZE + 1
        bar = tqdm(total=num_chunks, desc="Uploading", unit="MB")
        begin = 0
        for chunk in file_chunks(learnware_file):
            response = requests.post(url_upload, files={
                "chunk_file": chunk,
            }, data={
                "file_hash": file_hash,
                "chunk_begin": begin,
            }, headers=self.headers)

            result = response.json()

            if result['code'] != 0:
                raise Exception('upload failed: ' + json.dumps(result))
            
            begin += len(chunk)
            bar.update(1)
            pass
        bar.close()
        
        url_add = f"{self.host}/storage/add_learnware_uploaded"

        response = requests.post(url_add, json={
            "file_hash": file_hash,
            "semantic_specification": json.dumps(semantic_specification),
        }, headers=self.headers)

        result = response.json()

        if result['code'] != 0:
            raise Exception('upload failed: ' + json.dumps(result))
        
        return result['data']['learnware_id']

    
    def download_learnware(self, learnware_id, save_path):
        url = f"{self.host}/engine/download_learnware"

        response = requests.get(url, params={
            "learnware_id": learnware_id,
        }, headers=self.headers, stream=True)

        if response.status_code != 200:
            raise Exception('download failed: ' + json.dumps(response.json()))
        
        num_chunks = int(response.headers['Content-Length']) // CHUNK_SIZE + 1
        bar = tqdm(total=num_chunks, desc="Downloading", unit="MB")

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE): 
                f.write(chunk)
                bar.update(1)
                pass
            pass
        pass
    
    @require_login
    def list_learnware(self):
        url = f"{self.host}/user/list_learnware"
        response = requests.post(
            url, json={'page': 0, 'limit': 10000}, headers=self.headers)

        result = response.json()

        if result['code'] != 0:
            raise Exception('list failed: ' + json.dumps(result))
        
        learnware_list = result['data']['learnware_list']

        return learnware_list

    @require_login
    def search_learnware(self, specification: Specification):
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
                    url, files=files, 
                    data={"semantic_specification": json.dumps(specification.get_semantic_spec())},
                    headers=self.headers)
                
                result = response.json()

                if result['code'] != 0:
                    raise Exception('search failed: ' + json.dumps(result))
                
                for learnware in result['data']['learnware_list_single']:
                    returns.append({
                        "learnware_id": learnware['learnware_id'],
                        "semantic_specification": learnware['semantic_specification'],
                        "matching": learnware['matching'],
                    })
                    pass
                pass
            pass

        return returns

    @require_login
    def delete_learnware(self, learnware_id):
        url = f"{self.host}/user/delete_learnware"
        response = requests.post(
            url, json={'learnware_id': learnware_id}, headers=self.headers)
        
        result = response.json()

        if result['code'] != 0:
            raise Exception('delete failed: ' + json.dumps(result))
        pass
    pass