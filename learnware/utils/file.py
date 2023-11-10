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
