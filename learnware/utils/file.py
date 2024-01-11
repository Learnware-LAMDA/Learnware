import os
import zipfile

import yaml


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """save dict object into yaml file"""
    with open(save_path, "w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def read_yaml_to_dict(yaml_path: str):
    """load yaml file into dict object"""
    with open(yaml_path, "r") as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def convert_folder_to_zipfile(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w") as zip_obj:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zip_info = zipfile.ZipInfo(filename)
                zip_info.compress_type = zipfile.ZIP_STORED
                with open(file_path, "rb") as file:
                    zip_obj.writestr(zip_info, file.read())
