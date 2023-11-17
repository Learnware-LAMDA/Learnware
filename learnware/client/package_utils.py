import os
import re
import json
import yaml
import tempfile
import subprocess
from typing import List, Tuple
from . import utils


from ..logger import get_module_logger

logger = get_module_logger("package_utils")


def try_to_run(args, timeout=5, retry=5):
    sucess = False
    for i in range(retry):
        try:
            utils.system_execute(args=args, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sucess = True
            break
        except subprocess.TimeoutExpired as e:
            pass

    if not sucess:
        raise subprocess.TimeoutExpired(args, timeout)


def parse_pip_requirement(line: str):
    """Parse pip requirement line to package name"""

    line = line.strip()

    if len(line) == 0:
        return None

    if line[0] in ("#", "-"):
        return None

    package_str = line
    for split_ch in ("=", ">", "<", "!", "~", " "):
        split_ch_index = package_str.find(split_ch)
        if split_ch_index != -1:
            package_str = package_str[:split_ch_index]

    return package_str


def read_pip_packages_from_requirements(requirements_file: str) -> List[str]:
    """Read requiremnts.txt and parse it to list"""

    packages = []
    lines = []
    with open(requirements_file, "r") as fin:
        for line in fin:
            package_str = parse_pip_requirement(line)
            packages.append(package_str)
            lines.append(line)

    return packages, lines


def filter_nonexist_pip_packages(packages: list) -> Tuple[List[str], List[str]]:
    """Filter non-exist pip requirements

    Returns
    -------
    Tuple[List[str], List[str]]
        exist_packages: list of exist packages
        nonexist_packages: list of non-exist packages
    """

    exist_packages = []
    nonexist_packages = []
    for package in packages:
        try:
            package_name = parse_pip_requirement(package)
            if package_name != "learnware":
                try_to_run(args=["pip", "index", "versions", package_name], timeout=5)
                exist_packages.append(package)
                continue
        except Exception as e:
            logger.error(e)
        nonexist_packages.append(package)

    return exist_packages, nonexist_packages


def filter_nonexist_conda_packages(packages: list) -> Tuple[List[str], List[str]]:
    """Filter non-exist conda requirements

    Returns
    -------
    Tuple[List[str], List[str]]
        exist_packages: list of exist packages
        nonexist_packages: list of non-exist packages
    """

    def _process_dependency(dependency):
        modified_dependency = re.sub(r"=+", "=", dependency)
        match = re.search(r"([^=]*=[^=]*)=", modified_dependency)
        return match.group(1) if match else modified_dependency

    def _dry_run(test_yaml):
        with tempfile.TemporaryDirectory(prefix="conda_filter_") as tempdir:
            test_yaml_file = os.path.join(tempdir, "environment.yaml")
            with open(test_yaml_file, "w") as fout:
                yaml.safe_dump(test_yaml, fout)

            command = f"conda env create --name env_test --file {test_yaml_file} --dry-run --json"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = result.stdout.strip()
            last_bracket = stdout.rfind("\n{")
            if last_bracket != -1:
                stdout = stdout[last_bracket:]
            return json.loads(stdout).get("bad_deps", [])

    org_yaml = {
        "channels": ["defaults"],
        "dependencies": packages,
    }
    bad_deps = _dry_run(org_yaml)
    if len(bad_deps) == 0:
        return packages, []

    exist_packages = []
    nonexist_packages = []
    bad_deps = set([package.replace("=", "") for package in bad_deps])
    for package in packages:
        if package.replace("=", "") in bad_deps:
            if package.startswith("python="):
                exist_packages.append(_process_dependency(package))
            else:
                nonexist_packages.append(package)
        else:
            exist_packages.append(package)
    logger.info(f"Filtered out {len(nonexist_packages)} non-exist conda dependencies.")

    if not any(package.startswith("python=") for package in exist_packages):
        exist_packages = ["python=3.8"] + exist_packages
    return exist_packages, nonexist_packages


def read_conda_packages_from_dict(env_desc: dict) -> Tuple[List[str], List[str]]:
    """Read conda packages

    Parameters
    ----------
    env_desc : dict
        Dict of environment description

    Returns
    -------
    Tuple[List[str], List[str]]
        conda_packages: list of conda packages
        pip_packages: list of pip packages
    """

    conda_packages = env_desc.get("dependencies")
    if conda_packages is None:
        conda_packages = []
        pip_packages = []
    else:
        pip_packages = []
        conda_packages_ = []
        for package in conda_packages:
            if isinstance(package, dict) and "pip" in package:
                pip_packages = package["pip"]
                # pip_packages = [parse_pip_requirement(line) for line in pip_packages]
            elif isinstance(package, str):
                conda_packages_.append(package)

        conda_packages = conda_packages_

    return conda_packages, pip_packages


def filter_nonexist_conda_packages_file(yaml_file: str, output_yaml_file: str):
    with open(yaml_file, "r") as fin:
        env_desc = yaml.safe_load(fin)

    conda_packages, pip_packages = read_conda_packages_from_dict(env_desc)

    conda_packages, nonexist_conda_packages = filter_nonexist_conda_packages(conda_packages)
    pip_packages, nonexist_pip_packages = filter_nonexist_pip_packages(pip_packages)

    env_desc["dependencies"] = conda_packages
    if len(pip_packages) > 0:
        env_desc["dependencies"].append({"pip": pip_packages})

    with open(output_yaml_file, "w") as fout:
        yaml.safe_dump(env_desc, fout)

    return conda_packages, pip_packages, nonexist_conda_packages, nonexist_pip_packages


def filter_nonexist_pip_packages_file(requirements_file: str, output_file: str):
    packages, lines = read_pip_packages_from_requirements(requirements_file)

    exist_packages, nonexist_packages = filter_nonexist_pip_packages(packages)

    exist_packages = set(exist_packages)

    with open(output_file, "w") as fout:
        for package, line in zip(packages, lines):
            if package is not None and package in exist_packages:
                fout.write(line + "\n")

    logger.info(f"exist packages: {packages}")
    return exist_packages, nonexist_packages
