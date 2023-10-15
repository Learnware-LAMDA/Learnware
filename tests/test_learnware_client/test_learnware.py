import os
import zipfile
import tempfile
from learnware.learnware import get_learnware_from_dirpath
from learnware.test import get_semantic_specification
from learnware.client.container import LearnwaresContainer
from learnware.market import EasyMarket

if __name__ == "__main__":
    semantic_specification = get_semantic_specification()

    zip_path = "rf_tic.zip"
    with tempfile.TemporaryDirectory(suffix="learnware") as tempdir:
        learnware_dirpath = os.path.join(tempdir, "test")
        with zipfile.ZipFile(zip_path, "r") as z_file:
            z_file.extractall(learnware_dirpath)
        learnware = get_learnware_from_dirpath(
            id="test", semantic_spec=semantic_specification, learnware_dirpath=learnware_dirpath
        )

        with LearnwaresContainer(learnware, zip_path) as env_container:
            learnware = env_container.get_learnwares_with_container()[0]
            EasyMarket.check_learnware(learnware)
