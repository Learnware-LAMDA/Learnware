import os
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from learnware.logger import get_module_logger
from config import *

logger = get_module_logger("base_table", level="INFO")


class Recorder:
    def __init__(self, headers=["Mean", "Std Dev"], formats=["{:.2f}", "{:.2f}"]):
        assert len(headers) == len(formats), "Headers and formats length must match."
        self.data = defaultdict(list)
        self.headers = headers
        self.formats = formats

    def record(self, user, scores):
        self.data[user].append(scores)

    def get_performance_data(self, user):
        return self.data.get(user, [])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4, default=list)

    def load(self, path):
        with open(path, "r") as f:
            self.data = json.load(f, object_hook=lambda x: defaultdict(list, x))
    
    def should_test_method(self, user, idx, path):
        if os.path.exists(path):
            self.load(path)
            return user not in self.data or idx > len(self.data[user]) - 1
        return True


def process_single_aug(user, idx, scores, recorders, root_path):
    try:
        scores_array = np.array(scores)
        while scores_array.ndim < 3:
            scores_array = scores_array[np.newaxis, :]
        select_scores = scores_array[:, 0, :].tolist()
        mean_scores = np.mean(scores_array, axis=1).tolist()
        oracle_scores = np.min(scores_array, axis=1).tolist()

        for method_name, scores in zip(["select_score", "mean_score", "oracle_score"], 
                                       [select_scores, mean_scores, oracle_scores]):
            recorders[method_name].record(user, scores)
            save_path = os.path.join(root_path, f"{method_name}_performance.json")
            recorders[method_name].save(save_path)
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error in process_single_aug for user {user}, idx {idx}: {error_message}")


def plot_performance_curves(user, recorders, task, n_labeled_list):
    plt.figure(figsize=(10, 6))
    plt.xticks(range(len(n_labeled_list)), n_labeled_list)
    
    for method, recorder in recorders.items():
        if method == "hetero_single_aug":
            continue
        
        scores_array = recorder.get_performance_data(user)
        if scores_array:
            mean_curve, std_curve = [], []
            for i in range(len(n_labeled_list)):
                sub_scores_array = np.vstack([lst[i] for lst in scores_array])
                sub_scores_mean = np.squeeze(np.mean(sub_scores_array, axis=0))
                mean_curve.append(np.mean(sub_scores_mean))
                std_curve.append(np.std(sub_scores_mean))
                
            mean_curve = np.array(mean_curve)
            std_curve = np.array(std_curve)

            method_plot = '_'.join(method.split('_')[1:]) if method not in ['user_model', 'oracle_score', 'select_score', 'mean_score'] else method
            style = styles.get(method_plot, {"color": "black", "linestyle": "-"})
            plt.plot(mean_curve, label=labels.get(method_plot), **style)

            plt.fill_between(
                range(len(mean_curve)), 
                mean_curve - std_curve, 
                mean_curve + std_curve, 
                color=style["color"], 
                alpha=0.2
            )

    plt.xlabel("Amount of Labeled User Data", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title(f"Results on Homo Table Experimental Scenario", fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    root_path = os.path.abspath(os.path.join(__file__, ".."))
    fig_path = os.path.join(root_path, "results", "figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, f"{task}_labeled_curves.svg"), bbox_inches="tight", dpi=700)