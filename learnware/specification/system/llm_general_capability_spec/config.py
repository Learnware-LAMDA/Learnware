from typing import List
import numpy as np

from ....tests.benchmarks import LLMBenchmarkConfig

# Score normalization functions, copied from the interactive notebook in https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization 

def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100

def compute_bbh_score(data):
    bbh_subtasks = {
        "sports_understanding": 2,
        "tracking_shuffled_objects_three_objects": 3,
        "navigate": 2,
        "snarks": 2,
        "date_understanding": 6,
        "reasoning_about_colored_objects": 18,
        "object_counting": 19,
        "logical_deduction_seven_objects": 7,
        "geometric_shapes": 11,
        "web_of_lies": 2,
        "movie_recommendation": 6,
        "logical_deduction_five_objects": 5,
        "salient_translation_error_detection": 6,
        "disambiguation_qa": 3,
        "temporal_sequences": 4,
        "hyperbaton": 2,
        "logical_deduction_three_objects": 3,
        "causal_judgement": 2,
        "formal_fallacies": 2,
        "tracking_shuffled_objects_seven_objects": 7,
        "ruin_names": 6,
        "penguins_in_a_table": 5,
        "boolean_expressions": 2,
        "tracking_shuffled_objects_five_objects": 5
    }
    # Normalize BBH subtasks scores
    bbh_scores = []
    for subtask, num_choices in bbh_subtasks.items():
        subtask_key = f'leaderboard_bbh_{subtask}'
        if subtask_key in data['results']:
            bbh_raw_score = data['results'][subtask_key]['acc_norm,none']
            lower_bound = 1 / num_choices
            normalized_score = normalize_within_range(bbh_raw_score, lower_bound, 1.0)
            bbh_scores.append(normalized_score)

    # Average BBH score
    bbh_score = sum(bbh_scores) / len(bbh_scores)
    return round(bbh_score, 2)

def compute_gpqa_score(data):
    gpqa_subtasks = [
      "leaderboard_gpqa_diamond",
      "leaderboard_gpqa_extended",
      "leaderboard_gpqa_main"
    ]
    # Normalize GPQA scores
    gpqa_raw_scores = []
    for subtask in gpqa_subtasks:
        gpqa_raw_scores.append(data['results'][subtask]['acc_norm,none'])
    gpqa_raw_score = sum(gpqa_raw_scores) / len(gpqa_raw_scores)
    gpqa_score = normalize_within_range(gpqa_raw_score, 0.25, 1.0)
    return round(gpqa_score, 2)

def compute_ifeval_score(data):
    # Compute IFEval
    ifeval_inst_score = data['results']['leaderboard_ifeval']['inst_level_strict_acc,none'] * 100
    ifeval_prompt_score = data['results']['leaderboard_ifeval']['prompt_level_strict_acc,none'] * 100

    # Average IFEval scores
    ifeval_score = (ifeval_inst_score + ifeval_prompt_score) / 2
    return round(ifeval_score, 2)

def compute_math_score(data):
    math_subtasks =  [
      "leaderboard_math_algebra_hard",
      "leaderboard_math_counting_and_prob_hard",
      "leaderboard_math_geometry_hard",
      "leaderboard_math_intermediate_algebra_hard",
      "leaderboard_math_num_theory_hard",
      "leaderboard_math_prealgebra_hard",
      "leaderboard_math_precalculus_hard"
    ]
    # Calculate the MATH score
    math_raw_scores = []
    for subtask in math_subtasks:
        math_raw_scores.append(data['results'][subtask]['exact_match,none'])
    math_raw_score = sum(math_raw_scores) / len(math_raw_scores)
    math_score = normalize_within_range(math_raw_score, 0, 1.0)
    return round(math_score, 2)

def compute_mmlu_pro_score(data):
    # Normalize MMLU PRO scores
    mmlu_pro_raw_score = data['results']['leaderboard_mmlu_pro']['acc,none']
    mmlu_pro_score = normalize_within_range(mmlu_pro_raw_score, 0.1, 1.0)
    return round(mmlu_pro_score, 2)

def compute_musr_score(data):
    musr_subtasks = {
        'murder_mysteries': 2,
        'object_placements': 5,
        'team_allocation': 3
    }
    # Normalize MUSR scores
    musr_scores = []

    for subtask, num_choices in musr_subtasks.items():
        musr_raw_score = data['results'][f'leaderboard_musr_{subtask}']['acc_norm,none']
        lower_bound = 1 / num_choices
        normalized_score = normalize_within_range(musr_raw_score, lower_bound, 1.0)
        musr_scores.append(normalized_score)

    musr_score = sum(musr_scores) / len(musr_scores)
    return round(musr_score, 2)


test_benchmark_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="mmlu_anatomy",
        dataset_path="hails/mmlu_no_train",
        validation_split="validation",
        test_split="test",
        eval_metric="acc",
    ),
]

general_capability_benchmark_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="leaderboard_bbh",
        dataset_path="SaylorTwift/bbh",
        test_split="test",
        score_function=compute_bbh_score,
    ),
    LLMBenchmarkConfig(
        name="leaderboard_gpqa",
        dataset_path="Idavidrein/gpqa",
        test_split="train",
        score_function=compute_gpqa_score,
    ),
    LLMBenchmarkConfig(
        name="leaderboard_ifeval",
        dataset_path="wis-k/instruction-following-eval",
        test_split="train",
        score_function=compute_ifeval_score,
    ),
    LLMBenchmarkConfig(
        name="leaderboard_math_hard",
        dataset_path="lighteval/MATH-Hard",
        train_split="train",
        test_split="test",
        score_function=compute_math_score,
    ),
    LLMBenchmarkConfig(
        name="leaderboard_mmlu_pro",
        dataset_path="TIGER-Lab/MMLU-Pro",
        validation_split="validation",
        test_split="test",
        score_function=compute_mmlu_pro_score,
    ),
    LLMBenchmarkConfig(
        name="leaderboard_musr",
        dataset_path="TAUR-Lab/MuSR",
        score_function=compute_musr_score,
    ),
]
