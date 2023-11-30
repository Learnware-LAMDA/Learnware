import os

import fire

from benchmarks.utils import build_learnware, build_specification, evaluate
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market, BaseUserInfo

PROXY_IP = "172.24.57.111"
os.environ["HTTP_PROXY"] = "http://"+PROXY_IP+":7890"
os.environ["HTTPS_PROXY"] = "http://"+PROXY_IP+":7890"

class CifarDatasetWorkflow:

    def prepare_learnware(self, market_size=30, rebuild=False):
        """initialize learnware market"""
        # learnware.init()

        market = instantiate_learnware_market(name="easy", market_id="dataset_cifar_workflow", rebuild=rebuild)

        for i in range(market_size - len(market)):
            print("=" * 20 + "learnware {}".format(i) + "=" * 20)
            build_learnware("cifar10", market)

        print("Total Item:", len(market))

    def evaluate(self, user_size=20):
        market = instantiate_learnware_market(name="easy", market_id="dataset_cifar_workflow", rebuild=False)

        for i in range(user_size):
            user_spec, dataset = build_specification("cifar10", i)

            user_info = BaseUserInfo(semantic_spec=LearnwareClient.create_semantic_specification(
                    self=None,
                    description="For Cifar Dataset Workflow",
                    data_type="Image",
                    task_type="Classification",
                    library_type="PyTorch",
                    scenarios=["Computer"],
                    output_description={"Dimension": 10, "Description": {str(i): "i" for i in range(10)}}),
                stat_info={"RKMEImageSpecification": user_spec})

            search_result = market.search_learnware(user_info)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()

            loss_list = []
            for single_item in single_result[:3]:
                loss, acc = evaluate(single_item.learnware, dataset)
                loss_list.append(loss)

            print(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, loss: {loss_list[0]}"
            )


if __name__ == "__main__":
    fire.Fire(CifarDatasetWorkflow)
