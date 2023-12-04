import os

import fire
import numpy as np
from numpy import mean
from torch.utils.data import DataLoader

import learnware
from benchmarks.utils import build_learnware, build_specification, evaluate, Recorder
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser
from learnware.specification import generate_rkme_image_spec

PROXY_IP = "172.24.57.111"
os.environ["HTTP_PROXY"] = "http://"+PROXY_IP+":7890"
os.environ["HTTPS_PROXY"] = "http://"+PROXY_IP+":7890"

class CifarDatasetWorkflow:

    def prepare_learnware(self, market_size=50, market_id=None, rebuild=False):
        """initialize learnware market"""
        learnware.init()
        assert not rebuild

        market_id = "dataset_cifar_workflow" if market_id is None else market_id
        orders = np.stack([np.random.permutation(10) for _ in range(market_size)])

        print("Using market_id", market_id)
        market = instantiate_learnware_market(name="easy", market_id=market_id, rebuild=rebuild)

        for i, order in enumerate(orders[len(market):]):
            print("=" * 20 + "learnware {}".format(i) + "=" * 20)
            print("order:", order)
            build_learnware("cifar10", market, order)

        print("Total Item:", len(market))

    def evaluate_unlabeled(self, user_size=100, market_id=None):
        learnware.init()

        market_id = "dataset_cifar_workflow" if market_id is None else market_id
        orders = np.stack([np.random.permutation(10) for _ in range(user_size)])

        print("Using market_id", market_id)
        market = instantiate_learnware_market(name="easy", market_id=market_id, rebuild=False)

        top_1_acc_record, ensemble_acc_record, best_acc_record, mean_acc_record = [], [], [], []
        top_1_loss_record, ensemble_loss_record, best_loss_record, mean_loss_record = [], [], [], []

        recorder = Recorder()
        for i, order in enumerate(orders):
            print("=" * 20 + "user {}".format(i) + "=" * 20)
            print("order:", order)
            user_spec, dataset = build_specification("cifar10", i, order)

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

            loss_list, acc_list = [], []
            for item in market.get_learnwares():
                loss, acc = evaluate(item, dataset)
                loss_list.append(loss)
                acc_list.append(acc)
            recorder.record("Best", accuracy=max(acc_list), loss=min(loss_list))
            recorder.record("Average", accuracy=mean(acc_list), loss=mean(loss_list))

            top_1_loss, top_1_acc  = evaluate(single_result[0].learnware, dataset)
            recorder.record("Top-1 Learnware", accuracy=top_1_acc, loss=top_1_loss)

            reuse_ensemble = AveragingReuser(learnware_list=multiple_result[0].learnwares, mode="vote_by_prob")
            # reuse_ensemble = AveragingReuser(learnware_list=[item.learnware for item in single_result[:3]], mode="vote_by_prob")
            ensemble_loss, ensemble_acc = evaluate(reuse_ensemble, dataset)
            recorder.record("Voting Reuse", accuracy=ensemble_acc, loss=ensemble_loss)

            reuse_job_selector = JobSelectorReuser(learnware_list=multiple_result[0].learnwares, use_herding=False)
            job_loss, job_acc = evaluate(reuse_job_selector, dataset)
            recorder.record("Job Selector", accuracy=job_acc, loss=job_loss)

            print(recorder.summary())


if __name__ == "__main__":
    fire.Fire(CifarDatasetWorkflow)
