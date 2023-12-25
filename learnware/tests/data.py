import json
import requests
from tqdm import tqdm

from ..config import C


class GetData:
    def __init__(self, host=None, chunk_size=1024 * 1024):
        self.headers = None

        if host is None:
            self.host = C.backend_host
        else:
            self.host = host

        self.chunk_size = chunk_size

    def download_file(self, file_path: str, save_path: str):
        url = f"{self.host}/datasets/download_datasets"

        response = requests.get(
            url,
            params={
                "dataset": file_path,
            },
            stream=True,
        )

        if response.status_code != 200:
            raise Exception("download failed: " + json.dumps(response.json()))

        num_chunks = int(response.headers["Content-Length"]) // self.chunk_size + 1
        bar = tqdm(total=num_chunks, desc="Downloading", unit="MB")

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                f.write(chunk)
                bar.update(1)
