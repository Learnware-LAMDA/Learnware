import os
import unittest
import tempfile
import logging

import learnware
from learnware.learnware import Learnware
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market, BaseUserInfo

learnware.init(logging_level=logging.WARNING)


class TestSearch(unittest.TestCase):
    client = LearnwareClient()

    @classmethod
    def setUpClass(cls):
        cls.market = instantiate_learnware_market(market_id="search_test", name="hetero", rebuild=True)
        if cls.client.is_connected():
            cls._build_learnware_market()

    @classmethod
    def _build_learnware_market(cls):
        table_learnware_ids = ["00001951", "00001980", "00001987"]
        image_learnware_ids = ["00000851", "00000858", "00000841"]
        text_learnware_ids = ["00000652", "00000637"]
        learnware_ids = table_learnware_ids + image_learnware_ids + text_learnware_ids
        with tempfile.TemporaryDirectory(prefix="learnware_search_test") as tempdir:
            for learnware_id in learnware_ids:
                learnware_zippath = os.path.join(tempdir, f"learnware_{learnware_id}.zip")
                try:
                    cls.client.download_learnware(learnware_id=learnware_id, save_path=learnware_zippath)
                    semantic_spec = (
                        cls.client.load_learnware(learnware_path=learnware_zippath)
                        .get_specification()
                        .get_semantic_spec()
                    )
                except Exception:
                    print("'learnware_id' is passed due to the network problem.")
                cls.market.add_learnware(
                    learnware_zippath,
                    learnware_id=learnware_id,
                    semantic_spec=semantic_spec,
                    checker_names=["EasySemanticChecker"],
                )

    def _skip_test(self):
        if not self.client.is_connected():
            print("Client can not connect!")
            return True
        return False

    def test_image_search(self):
        if not self._skip_test():
            learnware_id = "00000619"
            try:
                learnware: Learnware = self.client.load_learnware(learnware_id=learnware_id)
            except Exception:
                print("'test_image_search' is passed due to the network problem.")
            user_info = BaseUserInfo(stat_info=learnware.get_specification().get_stat_spec())
            search_result = self.market.search_learnware(user_info)
            print("Single Search Results:", search_result.get_single_results())
            print("Multiple Search Results:", search_result.get_multiple_results())

    def test_text_search(self):
        if not self._skip_test():
            learnware_id = "00000653"
            try:
                learnware: Learnware = self.client.load_learnware(learnware_id=learnware_id)
            except Exception:
                print("'test_text_search' is passed due to the network problem.")
            user_info = BaseUserInfo(stat_info=learnware.get_specification().get_stat_spec())
            search_result = self.market.search_learnware(user_info)
            print("Single Search Results:", search_result.get_single_results())
            print("Multiple Search Results:", search_result.get_multiple_results())

    def test_table_search(self):
        if not self._skip_test():
            learnware_id = "00001950"
            try:
                learnware: Learnware = self.client.load_learnware(learnware_id=learnware_id)
            except Exception:
                print("'test_table_search' is passed due to the network problem.")
            user_info = BaseUserInfo(stat_info=learnware.get_specification().get_stat_spec())
            search_result = self.market.search_learnware(user_info)
            print("Single Search Results:", search_result.get_single_results())
            print("Multiple Search Results:", search_result.get_multiple_results())


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestSearch("test_image_search"))
    _suite.addTest(TestSearch("test_text_search"))
    _suite.addTest(TestSearch("test_table_search"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
