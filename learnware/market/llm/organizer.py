import os
import tempfile
import zipfile
import traceback
from shutil import copyfile
from typing import List, Union

from ..heterogeneous import HeteroMapTableOrganizer
from ..base import BaseChecker
from ...config import C
from ...utils import read_yaml_to_dict, save_dict_to_yaml
from ...logger import get_module_logger
from ...specification import LLMGeneralCapabilitySpecification
from ...tests.benchmarks import BenchmarkConfig

logger = get_module_logger("llm_easy_organizer")


class LLMEasyOrganizer(HeteroMapTableOrganizer):

    def _update_learnware_general_capability_spec(
        self, ids: Union[str, List[str]]
    ):
        """Update learnware by ids, attempting to generate LLMGeneralCapabilitySpecification for them.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of target learnware
            List[str]: A list of ids of target learnwares
        """
        if isinstance(ids, str):
            ids = [ids]

        for idx in ids:
            try:
                general_capability_spec = LLMGeneralCapabilitySpecification()
                general_capability_spec.generate_stat_spec_from_system(learnware=self.learnware_list[idx])
                general_capability_spec_config = {
                    "module_path": "learnware.specification",
                    "class_name": general_capability_spec.type,
                    "file_name": "general_capability_spec.json",
                    "kwargs": {},
                }

                zip_path = self.learnware_zip_list[idx]
                folder_dir = self.learnware_folder_list[idx]
                self.learnware_list[idx].update_stat_spec(general_capability_spec.type, general_capability_spec)

                with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                    # update yaml file
                    with zipfile.ZipFile(zip_path, "r") as z_file:
                        z_file.extract(C.learnware_folder_config["yaml_file"], tempdir)

                    learnware_yaml_path = os.path.join(tempdir, C.learnware_folder_config["yaml_file"])
                    yaml_config = read_yaml_to_dict(learnware_yaml_path)
                    if "stat_specifications" in yaml_config:
                        yaml_config["stat_specifications"].append(general_capability_spec_config)
                    else:
                        yaml_config["stat_specifications"] = [general_capability_spec_config]
                        pass
                    save_dict_to_yaml(yaml_config, learnware_yaml_path)

                    with zipfile.ZipFile(zip_path, "a") as z_file:
                        z_file.write(learnware_yaml_path, C.learnware_folder_config["yaml_file"])

                    # save general capability specification
                    stat_spec_path = os.path.join(tempdir, general_capability_spec_config["file_name"])
                    general_capability_spec.save(stat_spec_path)
                    with zipfile.ZipFile(zip_path, "a") as z_file:
                        z_file.write(stat_spec_path, general_capability_spec_config["file_name"])

                    # update learnware folder
                    copyfile(learnware_yaml_path, os.path.join(folder_dir, C.learnware_folder_config["yaml_file"]))
                    copyfile(stat_spec_path, os.path.join(folder_dir, general_capability_spec_config["file_name"]))

            except Exception as err:
                traceback.print_exc()
                logger.warning(f"Learnware {idx} generate LLMGeneralCapabilitySpecification failed!")

    def _get_llm_base_model_learnware_ids(self, ids: Union[str, List[str]]) -> List[str]:
        """Get learnware ids that corresponding learnware contains a llm base model.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of target learnware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        List[str]
            Learnware ids
        """
        if isinstance(ids, str):
            ids = [ids]

        ret = []
        for idx in ids:
            semantic_spec = self.learnware_list[idx].get_specification().get_semantic_spec()
            if (
                semantic_spec["Data"]["Values"] == ["Text"]
                and semantic_spec["Task"]["Values"] == ["Text Generation"]
                and semantic_spec["Model"]["Values"] == ["Base Model"]
            ):
                ret.append(idx)
        return ret
