# LLM Dataset Workflow Example

## Introduction

This workflow refers to Section 5 of our paper "Learnware Retrieval with Parameter Vector Specification". We build three learnware dock systems of 8B-level LLMs across three domains: finance, healthcare, and mathematics. We evaluate them on public evaluation benchmarks.

We first train multiple models under different configurations by SFT on different datasets using LoRA. Qwen2.5-7B, Llama3.1-8B, Llama3.1-8B-Instruct are our base models. Then we generate specifications for each model and apply a retrieval algorithm to select the most suitable learnware based on user task requirements. The retrieved learnware is then evaluated on the corresponding task under the **Task-Level** evaluation setting using EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

We compare PAVE against several baselines, including the Random selection strategy, the Best-single model, base models used for fine-tuning, and well-known LLMs with over 70B parameters. Best-single refers to the model with the highest average score among the learnware candidates.

We do not distinguish between different models fine-tuned with the same instruction dataset, so if our method select a learnware for solving a given task, the performance is actually calculated by the average of all the models with the selected instruction dataset.

## Run the code

Since the evaluation of LLM is a time-consuming process, we provide our evaluation results of all models in a table to help you quickly get the final system performance.

Run the following command to get results using the performance  table of all models in medical/math/finance scenario (skip evaluation). **We recommend you to run these.**

```bash
python workflow.py llm_example medical
python workflow.py llm_example math
python workflow.py llm_example finance
```

Run the following command to obtain results for medical, mathematical, and financial scenarios (including evaluation). In the medical scenario, it takes 3-4 hours to get the final results on one A100 GPU. For math and finance scenario, the process is significantly more time-consuming and requires at least four A100 GPUs.

```bash
python workflow.py llm_example medical --skip_eval False
python workflow.py llm_example math --skip_eval False
python workflow.py llm_example finance --skip_eval False
```

Following [FinBen](https://github.com/The-FinAI/PIXIU), for evaluation in finance scenario, you need to first copy the folder ```extra_tasks/flare``` into the ```tasks``` directory within the installation path of ```lm_eval```. For example, run the following command:

```bash
cp -r extra_tasks/flare ~/anaconda3/envs/{env_name}/lib/python3.11/site-packages/lm_eval/tasks/
```

## Results

### Finance

The table below shows the performance value of different methods or language models in finance scenario.

| User                  | Qwen2.5-7B   | Llama3.1-8B-Instruct   | Llama3.1-8B   | Qwen1.5-110B   | Qwen2.5-72B   | Llama3.1-70B-Instruct   | Random   | Best-single   | PAVE   | Oracle   |
|:----------------------|:-------------|:-----------------------|:--------------|:---------------|:--------------|:------------------------|:---------|:--------------|:-------|:---------|
| australian            | 43.17        | 44.6                   | 43.17         | 43.17          | 43.17         | 47.48                   | 44.45    | 42.21         | 56.83  | 56.83    |
| cra_lendingclub       | 80.82        | 76.33                  | 57.34         | 80.82          | 47.01         | 53.07                   | 81.52    | 80.82         | 92.07  | 92.07    |
| fiqasa                | 38.3         | 40.43                  | 56.17         | 63.4           | 64.26         | 68.51                   | 46.53    | 32.06         | 76.38  | 76.38    |
| fpb                   | 76.08        | 32.78                  | 30.72         | 70.72          | 78.35         | 78.04                   | 67.95    | 77.73         | 84.25  | 84.25    |
| german                | 65.0         | 49.5                   | 66.0          | 66.0           | 66.5          | 43.5                    | 51.5     | 65.33         | 67.06  | 67.06    |
| headlines             | 74.81        | 59.95                  | 59.95         | 62.96          | 77.84         | 77.53                   | 72.43    | 95.61         | 95.61  | 95.61    |
| ner                   | 21.75        | 0.62                   | 9.01          | 17.89          | 9.36          | 9.52                    | 24.99    | 23.98         | 52.79  | 52.79    |
| sm_acl                | 51.1         | 51.4                   | 51.34         | 49.3           | 51.56         | 49.38                   | 51.42    | 50.71         | 52.82  | 53.63    |
| sm_bigdata            | 55.3         | 55.57                  | 52.79         | 51.02          | 50.27         | 47.76                   | 53.86    | 55.52         | 52.4   | 55.88    |
| sm_cikm               | 58.44        | 54.24                  | 54.07         | 44.01          | 58.27         | 47.86                   | 55.89    | 57.98         | 55.99  | 58.52    |
| causal20_sc           | 65.14        | 88.48                  | 79.45         | 83.75          | 76.17         | 87.16                   | 74.71    | 88.61         | 84.17  | 88.61    |
| finarg_ecc_arc        | 64.78        | 46.67                  | 60.0          | 62.32          | 63.04         | 44.64                   | 62.27    | 57.87         | 64.31  | 68.36    |
| finarg_ecc_auc        | 48.3         | 51.81                  | 49.85         | 55.01          | 61.71         | 65.02                   | 52.08    | 48.68         | 58.08  | 58.08    |
| fomc                  | 60.48        | 29.44                  | 34.68         | 58.47          | 57.66         | 66.13                   | 56.05    | 61.36         | 62.7   | 62.7     |
| ma                    | 79.2         | 56.4                   | 51.0          | 81.4           | 84.6          | 83.2                    | 73.64    | 79.27         | 79.81  | 79.81    |
| mlesg                 | 35.67        | 32.67                  | 20.0          | 34.67          | 38.67         | 42.33                   | 31.99    | 38.33         | 33.42  | 38.33    |
| multifin_en           | 60.99        | 31.32                  | 28.39         | 65.38          | 63.55         | 68.5                    | 54.96    | 58.61         | 63.46  | 63.46    |
| Avg.                  | 57.61        | 47.19                  | 47.29         | 58.25          | 58.35         | 57.63                   | 56.25    | 59.69         | 66.6   | 67.79    |
| Avg. rank             | 5.94         | 7.35                   | 7.82          | 5.94           | 4.71          | 5.24                    | 6.47     | 5.47          | 2.88   | 1.65     |
| PAVE (win/tie/loss)   | 13/0/4       | 15/0/2                 | 16/0/1        | 14/0/3         | 12/0/5        | 11/0/6                  | 16/0/1   | 12/1/4        | nan    | 0/11/6   |
| Oracle (win/tie/loss) | 17/0/0       | 17/0/0                 | 17/0/0        | 15/0/2         | 13/0/4        | 12/0/5                  | 17/0/0   | 14/3/0        | 6/11/0 | nan      |

Our method, PAVE, demonstrates strong performance across financial tasks, achieving the highest average score among all methods, delivering an nearly 14\% improvement compared with the best large-scale model Qwen2.5-72B. It ranks first among learnware retrieval methods in 13 out of 17 tasks, retrieves the optimal learnware (tied with Oracle) on 11 and outperforms all contenders in 8. 

These results shows that our system can match or surpass large-scale models with over 70B parameters under the Task-Level evaluation setting, while requiring only the memory for models under 8B efficiently.

### Medical

The table below shows the performance value of different methods or language models in medical scenario.

| User                  | Qwen2.5-7B   | Flan-PaLM-540B   | Random   | Best-single   | PAVE   | Oracle   |
|:----------------------|:-------------|:-----------------|:---------|:--------------|:-------|:---------|
| medmcqa               | 59.93        | 57.6             | 60.2     | 62.49         | 62.49  | 62.49    |
| medqa_4options        | 64.18        | 67.6             | 63.74    | 64.81         | 65.59  | 65.59    |
| anatomy               | 71.85        | 63.7             | 71.33    | 70.37         | 71.85  | 72.96    |
| clinical_knowledge    | 77.36        | 80.4             | 78.21    | 78.49         | 78.87  | 79.25    |
| college_biology       | 82.64        | 88.9             | 84.34    | 84.03         | 85.42  | 86.11    |
| college_medicine      | 69.36        | 76.3             | 69.02    | 68.79         | 69.36  | 69.94    |
| medical_genetics      | 87.0         | 75.0             | 86.95    | 89.0          | 87.0   | 89.0     |
| professional_medicine | 78.68        | 83.8             | 77.37    | 78.68         | 79.78  | 79.78    |
| pubmedqa              | 75.2         | 79.0             | 75.67    | 76.8          | 75.8   | 76.8     |
| Avg.                  | 74.02        | 74.7             | 74.09    | 74.83         | 75.13  | 75.77    |
| Avg. rank             | 4.44         | 2.67             | 4.89     | 3.56          | 2.56   | 1.67     |
| PAVE (win/tie/loss)   | 6/3/0        | 3/0/6            | 9/0/0    | 6/1/2         | nan    | 0/3/6    |
| Oracle (win/tie/loss) | 9/0/0        | 3/0/6            | 9/0/0    | 6/3/0         | 6/3/0  | nan      |

As shown, PAVE achieves the highest average score across 9 tasks, even surpassing the large-scale model Flan-PaLM-540B. This demonstrates that our system, leveraging multiple models with fewer than 8B parameters, can outperform a single large-scale model in task-specific scenarios. Among learnware retrieval methods, PAVE performs best in 7 out of 9 tasks, tied with Oracle in 6.

Furthermore, PAVE outperforming Best-single suggests that its effectiveness comes not from a single exceptionally strong model but from its retrieval mechanism and the collective strength of all candidate models.

### Math

The table below shows the performance value of different methods or language models in math scenario.

| User                          | Qwen2.5-7B   | Qwen1.5-110B   | Random   | Best-single   | PAVE   | Oracle   |
|:------------------------------|:-------------|:---------------|:---------|:--------------|:-------|:---------|
| agieval_aqua_rat              | 41.73        | 38.98          | 40.09    | 41.33         | 38.98  | 41.73    |
| agieval_gaokao_mathcloze      | 16.95        | 38.14          | 11.72    | 13.14         | 17.8   | 17.8     |
| agieval_gaokao_mathqa         | 49.86        | 77.78          | 50.35    | 51.0          | 51.57  | 53.42    |
| agieval_math                  | 19.8         | 19.3           | 20.15    | 18.5          | 20.6   | 28.4     |
| agieval_sat_math              | 55.91        | 57.27          | 55.3     | 57.5          | 57.27  | 57.5     |
| cmmlu_college_mathematics     | 45.71        | 47.62          | 49.36    | 48.58         | 52.38  | 52.38    |
| cmmlu_elementary_mathematics  | 65.65        | 77.83          | 64.49    | 65.0          | 66.96  | 67.18    |
| cmmlu_high_school_mathematics | 61.59        | 77.44          | 62.5     | 64.32         | 60.98  | 64.63    |
| gsm8k                         | 84.08        | 84.91          | 80.79    | 83.92         | 84.15  | 84.15    |
| mathqa                        | 43.32        | 48.07          | 41.51    | 46.28         | 41.41  | 46.28    |
| mgsm_native_cot_zh            | 66.4         | 68.8           | 67.64    | 68.8          | 73.6   | 73.6     |
| minerva_math                  | 40.16        | 47.9           | 37.4     | 41.23         | 36.48  | 45.12    |
| abstract_algebra              | 54.0         | 53.0           | 53.83    | 52.0          | 56.0   | 56.0     |
| college_mathematics           | 53.0         | 52.0           | 53.61    | 53.5          | 53.0   | 58.0     |
| elementary_mathematics        | 72.75        | 78.84          | 73.63    | 73.02         | 75.13  | 75.13    |
| high_school_mathematics       | 55.93        | 60.0           | 55.21    | 55.19         | 55.56  | 56.86    |
| Avg.                          | 51.68        | 57.99          | 51.1     | 52.08         | 52.62  | 54.89    |
| Avg. rank                     | 4.31         | 2.56           | 4.56     | 4.0           | 3.19   | 1.56     |
| PAVE (win/tie/loss)           | 10/1/5       | 5/2/9          | 11/0/5   | 10/0/6        | nan    | 0/6/10   |
| Oracle (win/tie/loss)         | 15/1/0       | 7/0/9          | 16/0/0   | 14/2/0        | 10/6/0 | nan      |

PAVE achieves optimal retrieval performance (tied with Oracle) in 10 out of 16 tasks and even outperforms all other contenders in 5. However, the large-scale model achieves the highest average score and even beats Oracle (which denotes the optimal performance using one of our 8B-level models). This is likely due to their strong reasoning abilities that lack in smaller models, rather than a shortcoming of our method, as evidenced by the minimal difference in the "win/tie/loss" of PAVE and Oracle on Qwen1.5-110B.
