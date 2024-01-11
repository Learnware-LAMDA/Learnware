import os
import zipfile

from .file import convert_folder_to_zipfile, read_yaml_to_dict, save_dict_to_yaml
from .gpu import allocate_cuda_idx, choose_device, setup_seed
from .import_utils import is_torch_available
from .module import get_module_by_module_path
from ..config import SystemType, get_platform


def zip_learnware_folder(path: str, output_name: str):
    with zipfile.ZipFile(output_name, "w") as zip_ref:
        for root, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith(".pyc") or os.path.islink(full_path):
                    continue
                zip_ref.write(full_path, arcname=os.path.relpath(full_path, path))
