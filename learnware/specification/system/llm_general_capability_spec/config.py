from typing import List

from ....tests.benchmarks import LLMBenchmarkConfig

general_capability_benchmark_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="mmlu",
        dataset_path="hails/mmlu_no_train",
        validation_split="validation",
        test_split="test",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_bbh",
        dataset_path="SaylorTwift/bbh",
        test_split="test",
        eval_metric="acc_norm",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_gpqa",
        dataset_path="Idavidrein/gpqa",
        test_split="train",
        eval_metric="acc_norm",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_ifeval",
        dataset_path="wis-k/instruction-following-eval",
        test_split="train",
        eval_metric="inst_level_strict_acc",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_math_hard",
        dataset_path="lighteval/MATH-Hard",
        train_split="train",
        test_split="test",
        eval_metric="exact_match",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_mmlu_pro",
        dataset_path="TIGER-Lab/MMLU-Pro",
        validation_split="validation",
        test_split="test",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="leaderboard_musr",
        dataset_path="TAUR-Lab/MuSR",
        eval_metric="acc_norm",
    ),
]
