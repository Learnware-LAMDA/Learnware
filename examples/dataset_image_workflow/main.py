import os
from datetime import datetime

import fire
import numpy as np
import tqdm
from numpy import mean
import torch
from torch.utils.data import DataLoader, TensorDataset

import learnware
from benchmarks.utils import *
from benchmarks.dataset.data import faster_train, uploader_data
from benchmarks.models.conv import ConvModel
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser
from learnware.utils import choose_device

PROXY_IP = "172.27.138.61"
os.environ["HTTP_PROXY"] = "http://" + PROXY_IP + ":7890"
os.environ["HTTPS_PROXY"] = "http://" + PROXY_IP + ":7890"


class CifarDatasetWorkflow:

    def prepare(self, market_size=50, market_id=None, rebuild=False, faster=True):
        """initialize learnware market"""
        learnware.init()
        assert not rebuild

        market_id = "dataset_cifar_workflow" if market_id is None else market_id
        orders = np.stack([np.random.permutation(10) for _ in range(market_size)])

        print("Using market_id", market_id)
        market = instantiate_learnware_market(name="easy", market_id=market_id, rebuild=rebuild)

        device = choose_device(0)
        if faster:
            faster_train(device)
        for i, order in enumerate(orders[len(market):]):
            print("=" * 20 + "learnware {}".format(len(market)) + "=" * 20)
            print("order:", order)
            build_learnware("cifar10", market, order, device=device)

        print("Total Item:", len(market))

    def evaluate(self, user_size=100, market_id=None, faster=True):
        learnware.init()

        market_id = "dataset_cifar_workflow" if market_id is None else market_id
        orders = np.stack([np.random.permutation(10) for _ in range(user_size)])

        print("Using market_id", market_id)
        market = instantiate_learnware_market(name="easy", market_id=market_id, rebuild=False)

        device = choose_device(0)
        if faster:
            faster_train(device)
        unlabeled = Recorder(["Accuracy", "Loss"], ["{:.3f}% ± {:.3f}%", "{:.3f} ± {:.3f}"])
        labeled = Recorder(["Training", "Pruning"], ["{:.3f}% ± {:.3f}%", "{:.3f}% ± {:.3f}%"])
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
            unlabeled.record("Best", max(acc_list), min(loss_list))
            unlabeled.record("Average", mean(acc_list), mean(loss_list))

            top_1_loss, top_1_acc = evaluate(single_result[0].learnware, dataset)
            unlabeled.record("Top-1 Learnware", top_1_acc, top_1_loss)

            reuse_ensemble = AveragingReuser(learnware_list=multiple_result[0].learnwares, mode="vote_by_prob")
            ensemble_loss, ensemble_acc = evaluate(reuse_ensemble, dataset)
            unlabeled.record("Voting Reuse", ensemble_acc, ensemble_loss)

            reuse_job_selector = JobSelectorReuser(learnware_list=multiple_result[0].learnwares, use_herding=False)
            job_loss, job_acc = evaluate(reuse_job_selector, dataset)
            unlabeled.record("Job Selector", job_acc, job_loss)

            train_set, valid_set, spec_set, order = uploader_data(order=order)
            for labeled_size in tqdm.tqdm([100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]):
                loader = DataLoader(train_set, batch_size=labeled_size, shuffle=True)
                X, y = next(iter(loader))

                sampled_dataset = TensorDataset(X, y)
                mode_save_path = os.path.abspath(os.path.join(__file__, "..", "cache", "model.pth"))
                model = ConvModel(channel=X.shape[1], im_size=(X.shape[2], X.shape[3]),
                                  n_random_features=10).to(device)
                train_model(model, sampled_dataset, sampled_dataset, mode_save_path,
                            epochs=35, batch_size=128, device=device, verbose=False)
                model.load_state_dict(torch.load(mode_save_path))
                _, train_acc = evaluate(model, dataset, distribution=True)

                ensemble_pruning = EnsemblePruningReuser(learnware_list=multiple_result[0].learnwares)
                ensemble_pruning.fit(val_X=X, val_y=y)
                _, pruning_acc = evaluate(ensemble_pruning, dataset, distribution=False)

                labeled.record("{:d}".format(labeled_size), train_acc, pruning_acc)

            print(unlabeled.summary())
            print(labeled.summary())

        # Save recorder
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(os.path.join(__file__, "..", "log", formatted_time))
        os.makedirs(log_dir, exist_ok=True)
        unlabeled.save(os.path.join(log_dir, "unlabeled.json"))
        labeled.save(os.path.join(log_dir, "labeled.json"))

    def plot(self, record_dir):
        unlabeled = Recorder(["Accuracy", "Loss"], ["{:.3f}% ± {:.3f}%", "{:.3f} ± {:.3f}"])
        labeled = Recorder(["Training", "Pruning"], ["{:.3f}% ± {:.3f}%", "{:.3f}% ± {:.3f}%"])

        unlabeled.load(os.path.join(record_dir, "unlabeled.json"))
        labeled.load(os.path.join(record_dir, "labeled.json"))

        plot_labeled_performance_curves("Image", labeled[0], labeled[1],
                                        [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000],
                                        save_path=os.path.abspath(os.path.join(__file__, "..", "labeled.png")))


if __name__ == "__main__":
    fire.Fire(CifarDatasetWorkflow)
