import fire
import time
import tempfile
import os
import pandas as pd
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import lm_eval
from lm_eval.models.huggingface import HFLM

from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import BaseUserInfo, instantiate_learnware_market
from learnware.specification import GenerativeModelSpecification

from benchmark import Benchmark
from eval_config import CONFIG

logger = get_module_logger("llm_workflow", level="INFO")


class LLMWorkflow:
    def _plot_radar_chart(self, benchmark_name, results_table):
        labels = list(results_table.index)
        if benchmark_name == "finance":
            column_split = [
                ["Learnware", "Qwen2.5-7B", "Llama3.1-8B-Instruct", "Llama3.1-8B"],
                ["Learnware", "Qwen1.5-110B", "Qwen2.5-72B", "Llama3.1-70B-Instruct"],
                ["Learnware", "Random", "Best-single", "Oracle"],
            ]
            YTICKS = [0.2, 0.4, 0.6, 0.8, 1.0]
            ylim = (0, 1.15)
            x_label_fontsize = 4.5
            labels = [
                "Australian",
                "LendingClub",
                "FiQA-SA",
                "FPB",
                "German",
                "Headlines",
                "NER",
                "ACL18",
                "BigData22",
                "CIKM18",
                "SC",
                "FinArg-ARC",
                "FinArg-ACC",
                "FOMC",
                "MA",
                "MLESG",
                "MultiFin",
            ]
        elif benchmark_name == "math":
            column_split = [
                ["Learnware", "Qwen2.5-7B"],
                ["Learnware", "Qwen1.5-110B"],
                ["Learnware", "Random", "Best-single", "Oracle"],
            ]
            YTICKS = [0.4, 0.6, 0.8, 1.0]
            ylim = (0.3, 1.3)
            x_label_fontsize = 5
        elif benchmark_name == "medical":
            column_split = [
                ["Learnware", "Qwen2.5-7B"],
                ["Learnware", "Flan-PaLM-540B"],
                ["Learnware", "Random", "Best-single", "Oracle"],
            ]
            YTICKS = [0.8, 0.9, 1.0]
            ylim = (0.75, 1.1)
            x_label_fontsize = 8

        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw=dict(polar=True))

        model_names = ["Learnware vs Base Model", "Learnware vs Large-scale Model", "Specialized SLMs"]

        colors = [
            np.array([0.9, 0.17, 0.31]),
            np.array([1.0, 0.49, 0.0]),
            np.array([0.19, 0.55, 0.91]),
            np.array([0.56, 0.74, 0.56]),
            np.array([0.66, 0.66, 0.66]),
        ]

        for i, (ax, model_name) in enumerate(zip(axes, model_names)):
            ax.set_xticks(angles[:-1])
            ax.set_yticks(YTICKS)
            ax.set_xticklabels(labels, fontsize=x_label_fontsize, rotation=30)
            ax.set_yticklabels([str(y) for y in YTICKS])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_title(model_name, pad=30)

            methods = column_split[i]

            for i, (method, color) in enumerate(zip(methods, colors[: len(methods)])):
                if i == 0:
                    zorder = 2
                else:
                    zorder = 1

                values = (results_table[method] / results_table["Oracle"]).tolist()
                values += values[:1]

                ax.plot(angles, values, color=color, linewidth=2, label=method, zorder=zorder)
                ax.fill(angles, values, color=color, alpha=0.1, zorder=zorder)

            ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.85, 0.9))

        plt.tight_layout()
        os.makedirs("results/figs", exist_ok=True)
        plt.savefig(f"results/figs/llm-{benchmark_name}.pdf")

    def _anlysis_table(self, benchmark_name, table, score_results):
        if benchmark_name == "finance":
            start_column_id = 7
        else:  # math / medical
            start_column_id = 3
        table = table[:-1]
        performance = table.melt(
            id_vars=["Dataset"], value_vars=table.columns[start_column_id:], var_name="Source_Config"
        )
        performance_extra = table.iloc[:, :start_column_id]
        performance = pd.concat(
            [performance, performance["Source_Config"].str.extract(r"(.+)-(\d+)").rename(columns={0: "Learnware"})],
            axis=1,
        )
        performance["Learnware"] = performance["Learnware"].apply(lambda s: s[:-1] if s[-1] == "-" else s)
        performance = performance.rename(columns={"Dataset": "User"})
        performance.drop(columns=[1], inplace=True)
        perf_merged = performance[["User", "Learnware", "value"]].groupby(["Learnware", "User"]).mean().reset_index()

        performance_extra = performance_extra.rename(columns={"Dataset": "User"})
        performance_extra = performance_extra.set_index("User")

        score_results = pd.DataFrame(score_results)
        score_results["Rank-PAVE"] = (
            score_results.groupby("User")["Similarity"].rank(method="min", ascending=False).astype(int) - 1
        )
        adaptation_info = pd.merge(score_results, perf_merged, on=["Learnware", "User"])
        random_value = (adaptation_info[["User", "value"]].groupby(["User"]).mean()).rename(columns={"value": "Random"})
        oracle_value = (adaptation_info[["User", "value"]].groupby(["User"]).max()).rename(columns={"value": "Oracle"})
        pave_value = (
            adaptation_info[adaptation_info["Rank-PAVE"] < 1][["User", "value"]].groupby(["User"]).mean()
        ).rename(columns={"value": "Learnware"})

        # Best-single
        perf_pivot = perf_merged.pivot(index="User", columns="Learnware", values="value")
        best_column = perf_pivot.mean().idxmax()
        best_single = perf_pivot[[best_column]].rename(columns={best_column: "Best-single"})

        adaptation_table = pd.concat([random_value, pave_value, best_single, oracle_value], axis=1)

        # join performance_extra
        adaptation_table = performance_extra.join(adaptation_table)

        # Avg Rank
        ranks = adaptation_table.rank(axis=1, method="min", ascending=False)
        avg_rank = ranks.mean()

        # PAVE win/tie/loss
        pave_scores = adaptation_table["Learnware"]
        win_tie_loss = {}

        for col in adaptation_table.columns:
            if col == "Learnware":
                continue
            win = (pave_scores > adaptation_table[col]).sum()
            tie = (pave_scores == adaptation_table[col]).sum()
            loss = (pave_scores < adaptation_table[col]).sum()
            win_tie_loss[col] = f"{win}/{tie}/{loss}"

        # Oracle win/tie/loss
        oracle_scores = adaptation_table["Oracle"]
        win_tie_loss_o = {}

        for col in adaptation_table.columns:
            if col == "Oracle":
                continue
            win = (oracle_scores > adaptation_table[col]).sum()
            tie = (oracle_scores == adaptation_table[col]).sum()
            loss = (oracle_scores < adaptation_table[col]).sum()
            win_tie_loss_o[col] = f"{win}/{tie}/{loss}"

        adaptation_table.loc["Avg."] = adaptation_table.mean()
        adaptation_table.loc["Avg. rank"] = avg_rank
        adaptation_table = adaptation_table.round(2)
        adaptation_table.loc["Learnware (win/tie/loss)"] = win_tie_loss
        adaptation_table.loc["Oracle (win/tie/loss)"] = win_tie_loss_o

        print(adaptation_table.to_markdown())
        os.makedirs("results/tables", exist_ok=True)
        adaptation_table.to_csv(f"results/tables/llm-{benchmark_name}.csv")

        return adaptation_table

    def _prepare_market(self, benchmark: Benchmark, rebuild=False):
        client = LearnwareClient()
        self.llm_benchmark = benchmark
        self.llm_market = instantiate_learnware_market(
            market_id=f"llm_{self.llm_benchmark.name}", name="llm", rebuild=rebuild
        )
        self.user_semantic = client.get_semantic_specification(self.llm_benchmark.learnware_ids[0])
        self.user_semantic["Name"]["Values"] = ""
        self.user_semantic["Description"]["Values"] = ""
        self.user_semantic["License"]["Values"] = ["Apache-2.0", "Others"]

        if len(self.llm_market) == 0 or rebuild is True:
            for learnware_id in self.llm_benchmark.learnware_ids:
                with tempfile.TemporaryDirectory(prefix="llm_benchmark_") as tempdir:
                    zip_path = os.path.join(tempdir, f"{learnware_id}.zip")
                    for i in range(20):
                        try:
                            semantic_spec = client.get_semantic_specification(learnware_id)
                            client.download_learnware(learnware_id, zip_path)
                            self.llm_market.add_learnware(zip_path, semantic_spec)
                            break
                        except Exception:
                            time.sleep(1)
                            continue

        logger.info("Total Item: %d" % (len(self.llm_market)))

    def build_specification_and_cache(self, name, saved_folder, benchmark: Benchmark):
        generative_spec = GenerativeModelSpecification()
        generative_spec_path = os.path.join(saved_folder, name, "generative.pth")

        os.makedirs(os.path.join(saved_folder, name), exist_ok=True)

        if os.path.exists(generative_spec_path):
            generative_spec.load(generative_spec_path)
        else:
            train_dataset = benchmark.get_user_dataset(name)
            generative_spec.generate_stat_spec_from_data(dataset=train_dataset)
            generative_spec.save(generative_spec_path)

        return generative_spec

    def _get_scores(self, benchmark_name, base_model: str, adapter_path, batch_size="auto"):
        benchmark_configs = CONFIG[benchmark_name]
        task_manager = lm_eval.tasks.TaskManager()
        task_names = [config.name for config in benchmark_configs]

        if benchmark_name == "medical":
            lm_obj = HFLM(pretrained=base_model, peft=adapter_path, batch_size=batch_size)
            results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=task_names,
                task_manager=task_manager,
            )
        else:
            results_dir = f"./eval_results/{benchmark_name}"
            adapter_id = adapter_path.split("/")[-2] if adapter_path else None
            task_names_str = ",".join(task_names)
            if adapter_path:
                os.system(
                    f"CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
                --model_args pretrained={base_model},peft={adapter_path} \
                --tasks {task_names_str} \
                --batch_size {batch_size} \
                --output_path ./eval_results/{benchmark_name}"
                )
            elif base_model in ["Qwen/Qwen1.5-110B", "Qwen/Qwen2.5-72B", "NousResearch/Meta-Llama-3.1-70B-Instruct"]:
                os.system(
                    f"CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 -m lm_eval --model hf \
                --model_args pretrained={base_model},parallelize=True \
                --tasks {task_names_str} \
                --batch_size {batch_size} \
                --output_path ./eval_results/{benchmark_name}"
                )
            else:
                os.system(
                    f"CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
                --model_args pretrained={base_model} \
                --tasks {task_names_str} \
                --batch_size {batch_size} \
                --output_path ./eval_results/{benchmark_name}"
                )

            if adapter_id:
                for dir_name in os.listdir(results_dir):
                    if adapter_id in dir_name:
                        results_dir_path = os.path.join(results_dir, dir_name)
                        results_path = os.path.join(results_dir_path, sorted(os.listdir(results_dir_path))[-1])
                        break
            else:
                for dir_name in os.listdir(results_dir):
                    if dir_name == base_model.replace("/", "__"):
                        results_dir_path = os.path.join(results_dir, dir_name)
                        results_path = os.path.join(results_dir_path, sorted(os.listdir(results_dir_path))[-1])
                        break

            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        score_list = []
        for config in benchmark_configs:
            score = results["results"][config.name][f"{config.eval_metric},none"] * 100
            score = round(score, 2)
            logger.info(f"Name: {config.name}, Score: {score}")
            score_list.append(score)

        return score_list

    def llm_example(self, benchmark_name, rebuild=False, skip_eval=True):
        benchmark = Benchmark(benchmark_name)
        self._prepare_market(benchmark, rebuild)
        user_names = benchmark.get_user_names()

        score_results = {"User": [], "Learnware": [], "Similarity": []}

        for name in user_names:
            title = "=" * 20 + name + "=" * 20
            print(title)

            generative_spec = self.build_specification_and_cache(name, "user_specs", benchmark)

            user_info = BaseUserInfo(
                semantic_spec=self.user_semantic, stat_info={"GenerativeModelSpecification": generative_spec}
            )
            logger.info(f"Searching Market for user: {name}")

            search_result = self.llm_market.search_learnware(user_info)
            single_result = search_result.get_single_results()

            scores = {}
            for result in single_result:
                learnware_name = result.learnware.specification.semantic_spec["Name"]["Values"]
                match = re.match(r"(.+)-(\d+)", learnware_name)
                dataset_name = match.group(1)
                scores[dataset_name] = result.score

            for k, v in scores.items():
                score_results["User"].append(name)
                score_results["Learnware"].append(k)
                score_results["Similarity"].append(v)

        if not skip_eval:
            all_learnwares_ids = self.llm_market.get_learnware_ids()
            if benchmark_name == "medical":
                performance_table = {
                    "Qwen2.5-7B": self._get_scores(benchmark_name, "Qwen/Qwen2.5-7B", None),
                    "Flan-PaLM-540B": [
                        57.60,
                        67.60,
                        63.70,
                        80.40,
                        88.90,
                        76.30,
                        75.00,
                        83.80,
                        79.00,
                    ],  # copied from Open Medical LLM Leaderboard
                }
            elif benchmark_name == "math":
                performance_table = {
                    "Qwen2.5-7B": self._get_scores(benchmark_name, "Qwen/Qwen2.5-7B", None),
                    "Qwen1.5-110B": self._get_scores(benchmark_name, "Qwen/Qwen1.5-110B", None),
                }
            elif benchmark_name == "finance":
                performance_table = {
                    "Qwen2.5-7B": self._get_scores(benchmark_name, "Qwen/Qwen2.5-7B", None),
                    "Llama3.1-8B-Instruct": self._get_scores(
                        benchmark_name, "NousResearch/Meta-Llama-3.1-8B-Instruct", None
                    ),
                    "Llama3.1-8B": self._get_scores(benchmark_name, "NousResearch/Meta-Llama-3.1-8B", None),
                    "Qwen1.5-110B": self._get_scores(benchmark_name, "Qwen/Qwen1.5-110B", None),
                    "Qwen2.5-72B": self._get_scores(benchmark_name, "Qwen/Qwen2.5-72B", None),
                    "Llama3.1-70B-Instruct": self._get_scores(
                        benchmark_name, "NousResearch/Meta-Llama-3.1-70B-Instruct", None
                    ),
                }

            for learnware_id in all_learnwares_ids:
                learnware = self.llm_market.get_learnware_by_ids(learnware_id)
                base_model = learnware.specification.semantic_spec["Description"]["Values"].split(" ")[-1]
                adapter_path = os.path.join(self.llm_market.get_learnware_dir_path_by_ids(learnware_id), "adapter")
                score_list = self._get_scores(benchmark_name, base_model, adapter_path)
                performance_table[learnware.specification.semantic_spec["Name"]["Values"]] = score_list

            performance_table = pd.DataFrame(performance_table)
            performance_table = performance_table._append(performance_table.mean().round(2), ignore_index=True)
            datasets = benchmark.get_user_names()
            performance_table.insert(0, "Dataset", datasets + ["Avg"])
            performance_table.to_csv(f"model_performance/{benchmark_name}-new.csv", index=False)
        else:
            performance_table = pd.read_csv(f"model_performance/{benchmark_name}.csv")

        results_table = self._anlysis_table(benchmark_name, performance_table, score_results)
        self._plot_radar_chart(benchmark_name, results_table[:-4])


if __name__ == "__main__":
    fire.Fire(LLMWorkflow)
