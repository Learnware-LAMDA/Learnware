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
        self.data = defaultdict(lambda: defaultdict(list))
        self.headers = headers
        self.formats = formats

    def record(self, user, idx, scores):
        self.data[user][idx].append(scores)

    def get_performance_data(self, user):
        if user in self.data:
            return [idx_scores for idx_scores in self.data[user].values()]
        else:
            return []

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4, default=list)

    def load(self, path):
        with open(path, "r") as f:
            self.data = json.load(f, object_hook=lambda x: defaultdict(list, x))
    
    def should_test_method(self, user, idx, path):
        if os.path.exists(path):
            self.load(path)
            return user not in self.data or str(idx) not in self.data[user]
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
            recorders[method_name].record(user, idx, scores)
            save_path = os.path.join(root_path, f"{method_name}_performance.json")
            recorders[method_name].save(save_path)
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error in process_single_aug for user {user}, idx {idx}: {error_message}")


def analyze_performance(user, recorders):
    oracle_score_list = recorders["hetero_oracle_score"].get_performance_data(user)
    select_score_list = recorders["hetero_select_score"].get_performance_data(user)
    multi_avg_score_list = recorders["hetero_multiple_avg"].get_performance_data(user)
    mean_differences = {}

    for user_id in range(len(oracle_score_list)):
        select_scores = select_score_list[user_id]
        oracle_scores = oracle_score_list[user_id]
        mean_difference = np.mean(select_scores) - np.mean(oracle_scores)
        mean_differences[user_id] = mean_difference

    sorted_user_ids = sorted(mean_differences, key=mean_differences.get, reverse=True)

    for user_id in sorted_user_ids:
        single_multi_diff = np.mean(select_score_list[user_id]) - np.mean(multi_avg_score_list[user_id])
        logger.info(f"{user}, {user_id}, {mean_differences[user_id]}, {single_multi_diff}")


def plot_performance_curves(user, recorders, task, n_labeled_list):
    plt.figure(figsize=(10, 6))
    
    for method, recorder in recorders.items():
        if method == "hetero_single_aug":
            continue
        
        user_data = recorder.get_performance_data(user)
        
        if user_data:
            scores_array = np.array([np.array(lst) for lst in user_data])
            mean_scores = np.squeeze(np.mean(scores_array, axis=0))
            std_scores = np.squeeze(np.std(scores_array, axis=0))

            method_plot = '_'.join(method.split('_')[1:]) if method not in ['user_model', 'oracle_score', 'select_score', 'mean_score'] else method
            style = styles.get(method_plot, {"color": "black", "linestyle": "-"})
            plt.plot(range(len(n_labeled_list)), mean_scores, label=labels.get(method_plot), **style)

            std_scale = 0.2 if task == "Hetero" else 0.5
            plt.fill_between(range(len(n_labeled_list)), mean_scores - std_scale * std_scores, mean_scores + std_scale * std_scores, color=style["color"], alpha=0.2)

    plt.xticks(range(len(n_labeled_list)), n_labeled_list)
    plt.xlabel('Sample Size')
    plt.ylabel('RMSE')
    plt.title(f'Table {task} Limited Labeled Data')
    plt.legend()
    plt.tight_layout()
    
    root_path = os.path.abspath(os.path.join(__file__, ".."))
    fig_path = os.path.join(root_path, "results", "figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, f"{user}_labeled_{list(recorders.keys())}.svg"), bbox_inches="tight", dpi=700)