from typing import List

from learnware.tests.benchmarks import LLMBenchmarkConfig


medical_eval_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="medmcqa",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="medqa_4options",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_anatomy",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_clinical_knowledge",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_college_biology",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_college_medicine",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_medical_genetics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_professional_medicine",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="pubmedqa",
        eval_metric="acc",
    ),
]

math_eval_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="agieval_aqua_rat",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="agieval_gaokao_mathcloze",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="agieval_gaokao_mathqa",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="agieval_math",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="agieval_sat_math",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="cmmlu_college_mathematics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="cmmlu_elementary_mathematics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="cmmlu_high_school_mathematics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="gsm8k",
        eval_metric="exact_match,flexible-extract",
    ),
    LLMBenchmarkConfig(
        name="mathqa",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mgsm_native_cot_zh",
        eval_metric="exact_match,flexible-extract",
    ),
    LLMBenchmarkConfig(
        name="minerva_math",
        eval_metric="exact_match",
    ),
    LLMBenchmarkConfig(
        name="mmlu_abstract_algebra",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_college_mathematics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_elementary_mathematics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_high_school_mathematics",
        eval_metric="acc",
    ),
]

finance_eval_configs: List[LLMBenchmarkConfig] = [
    LLMBenchmarkConfig(
        name="australian",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="cra_lendingclub",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="fiqasa",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="fpb",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_clinical_knowledge",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_college_biology",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_college_medicine",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_medical_genetics",
        eval_metric="acc",
    ),
    LLMBenchmarkConfig(
        name="mmlu_professional_medicine",
        eval_metric="acc",
    ),
]

eval_configs = {
    "medical": medical_eval_configs,
    "math": math_eval_configs,
    "finance": finance_eval_configs
}