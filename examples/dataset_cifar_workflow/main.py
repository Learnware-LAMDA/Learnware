import os

import fire

from examples.dataset_cifar_workflow.benchmarks.utils import build_learnware, build_specification
from learnware.market import instantiate_learnware_market

PROXY_IP = "172.24.57.111"
os.environ["HTTP_PROXY"] = "http://"+PROXY_IP+":7890"
os.environ["HTTPS_PROXY"] = "http://"+PROXY_IP+":7890"

class CifarDatasetWorkflow:

    def prepare_learnware(self, market_size=30, rebuild=False):
        """initialize learnware market"""
        # learnware.init()

        market = instantiate_learnware_market(name="easy", market_id="dataset_cifar_workflow", rebuild=rebuild)

        for i in range(market_size - len(market)):
            build_learnware("cifar10", market)

        print("Total Item:", len(market))

    def evaluate(self, user_size=20):
        market = instantiate_learnware_market(name="easy", market_id="dataset_cifar_workflow", rebuild=rebuild)

        # for i in range(user_size):
        #     build_specification()


if __name__ == "__main__":
    fire.Fire(CifarDatasetWorkflow)
