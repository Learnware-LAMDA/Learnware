import os

import fire
import numpy as np
from numpy import mean
from torch.utils.data import DataLoader

import learnware
from benchmarks.utils import build_learnware, build_specification, evaluate, Recorder
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import AveragingReuser
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

    def evaluate(self, user_size=100, market_id=None):
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

            best_acc_record.append(max(acc_list))
            best_loss_record.append(min(loss_list))
            print("Best Accuracy: {:.3f}% ({:.3f}%), Best Loss: {:.3f} ({:.3f})".format(
                max(acc_list), mean(best_acc_record), min(loss_list), mean(best_loss_record)))
            recorder.record("Best", accuracy=max(acc_list), loss=min(loss_list))

            mean_acc_record.append(mean(acc_list))
            mean_loss_record.append(mean(loss_list))
            print("Avg Accuracy: {:.3f}% ({:.3f}%), Avg Loss: {:.3f} ({:.3f})".format(
                mean(acc_list), mean(mean_acc_record), mean(loss_list), mean(mean_loss_record)))
            recorder.record("Average", accuracy=mean(acc_list), loss=mean(loss_list))

            top_1_loss, top_1_acc  = evaluate(single_result[0].learnware, dataset)
            top_1_acc_record.append(top_1_acc)
            top_1_loss_record.append(top_1_loss)
            print(
                "Top-1\tAccuracy: {:.3f}% ({:.3f}%), Loss: {:.3f}({:.3f})".format(
                    top_1_acc, mean(top_1_acc_record), top_1_loss, mean(top_1_loss_record))
            )
            recorder.record("Top-1", accuracy=top_1_acc, loss=top_1_loss)

            reuse_ensemble = AveragingReuser(learnware_list=multiple_result[0].learnwares, mode="vote_by_prob")
            # reuse_ensemble = AveragingReuser(learnware_list=[item.learnware for item in single_result[:3]], mode="vote_by_prob")
            ensemble_loss, ensemble_acc = evaluate(reuse_ensemble, dataset)
            ensemble_acc_record.append(ensemble_acc)
            ensemble_loss_record.append(ensemble_loss)
            print(
                "Averaging Reuse\tAccuracy: {:.3f}% ({:.3f}%), Loss: {:.3f} ({:.3f})".format(
                    ensemble_acc, mean(ensemble_acc_record), ensemble_loss, mean(ensemble_loss_record))
            )
            recorder.record("Voting Reuse", accuracy=ensemble_acc, loss=ensemble_loss)

            print(recorder.latest())
            print(recorder.accumulated())


if __name__ == "__main__":
    fire.Fire(CifarDatasetWorkflow)
