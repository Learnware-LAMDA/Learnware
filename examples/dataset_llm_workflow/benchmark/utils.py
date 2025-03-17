import re
import random
from datasets import load_dataset, concatenate_datasets  
from typing import List

from .config import LEARNWARE_FIN, LEARNWARE_MATH, LEARNWARE_MED, USER_FIN


def preprocess_alpaca(docs):
    alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    instructions = docs["instruction"]
    inputs = docs["input"]
    outputs = docs["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return texts


def preprocess_alpaca_no_label(docs):
    alpaca_no_label_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
    instructions = docs["instruction"]
    inputs = docs["input"]
    texts = []
    for instruction, input in zip(instructions, inputs):
        text = alpaca_no_label_prompt.format(instruction, input)
        texts.append(text)
    return texts


def preprocess_alpaca_no_input(docs):
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["instruction"]
    outputs = docs["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_alpaca_no_input_no_label(docs):
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["instruction"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qr(docs):
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query"]
    outputs = docs["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qr_no_label(docs):
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["query"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qr_zh(docs):
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query_zh"]
    outputs = docs["response_zh"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qr_zh_no_label(docs):
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["query_zh"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qa(docs):
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qa_no_label(docs):
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["question"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qa_zh(docs):
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question_zh"]
    outputs = docs["answer_zh"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qa_zh_no_label(docs) -> str:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["question_zh"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_finance(docs) -> str:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        instruction.rstrip(' Answer:')
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_math_train(docs) -> str:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question"]
    outputs = docs["answer_detail"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


# Copied from Master
def preprocess_medmcqa(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def preprocess_medmcqa_val(docs):
    opas = docs["opa"]
    opbs = docs["opb"]
    opcs = docs["opc"]
    opds = docs["opd"]
    questions = docs["question"]
    option_ids = docs["cop"]
    texts = []
    for opa, opb, opc, opd, question, option_id in zip(opas, opbs, opcs, opds, questions, option_ids):
        option_choices = {
            "A": opa,
            "B": opb,
            "C": opc,
            "D": opd,
        }
        prompt = "Question: " + question + "\nChoices:\n"
        for choice, option in option_choices.items():
            prompt += f"{choice.upper()}. {option}\n"
        prompt += f"Answer: {list(option_choices.keys())[option_id]}"
        texts.append(prompt)
    return texts


def preprocess_medqa(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['sent1']}\n{answers}Answer:"


def preprocess_medqa_val(docs):
    ending0s = docs["ending0"]
    ending1s = docs["ending1"]
    ending2s = docs["ending2"]
    ending3s = docs["ending3"]
    sent1s = docs["sent1"]
    labels = docs["label"]
    texts = []
    for sent1, ending0, ending1, ending2, ending3, label in zip(sent1s, ending0s, ending1s, ending2s, ending3s, labels):
        option_choices = {
            "A": ending0,
            "B": ending1,
            "C": ending2,
            "D": ending3,
        }
        answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
        texts.append(f"Question: {sent1}\n{answers}Answer: {list(option_choices.keys())[label]}")
    return texts


def preprocess_mmlu(doc) -> str:
    question = doc["question"].strip()
    choices = doc["choices"]
    return "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:".format(
        question,
        choices[0],
        choices[1],
        choices[2],
        choices[3]
    )


def preprocess_mmlu_val(docs):
    questions = docs["question"]
    choices = docs["choices"]
    answers =  docs["answer"]
    texts = []
    for question, options, answer in zip(questions, choices, answers):
        texts.append(
            "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer: {}".format(
                question.strip(),
                options[0],
                options[1],
                options[2],
                options[3],
                ["A", "B", "C", "D"][answer]
            )
        )
    return texts


def preprocess_pubmedqa(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs,
        doc["QUESTION"],
    )


def preprocess_pubmedqa_val(docs):
    contexts_list = docs["CONTEXTS"]
    questions = docs["QUESTION"]
    answers = docs["final_decision"]
    texts = []
    for contexts, question, answer in zip(contexts_list, questions, answers):
        ctxs = "\n".join(contexts)
        texts.append("Abstract: {}\nQuestion: {}\nAnswer: {}".format(ctxs, question, answer))
    return texts


def preprocess_agieval(doc) -> str:
    return doc["query"]


def preprocess_cmmlu(doc) -> str:
    question = doc["Question"].strip()
    return "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：".format(
        question,
        doc["A"],
        doc["B"],
        doc["C"],
        doc["D"]
    )


def preprocess_cmmlu_val(docs):
    questions = docs["Question"]
    as_ = docs["A"]
    bs = docs["B"]
    cs = docs["C"]
    ds = docs["D"]
    answers =  docs["Answer"]
    texts = []
    for question, a, b, c, d, answer in zip(questions, as_, bs, cs, ds, answers):
        texts.append("{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：{}".format(
            question.strip(), a, b, c, d, answer
        ))
    return texts


def preprocess_mathqa(doc) -> str:
    return "Question: {}\nAnswer:".format(
        doc["Problem"]
    )


def preprocess_mgsm(doc) -> str:
    return "问题: "+doc["question"]+"\n逐步解答:"


def preprocess_gsm8k(doc) -> str:
    return "Question: {}\nAnswer:".format(doc["question"])


def preprocess_mathqa_val(docs):
    problems = docs["Problem"]
    corrects = docs["correct"]
    options = docs["options"]
    texts = []
    for problem, correct, option in zip(problems, corrects, options):
        choices = [
            c[4:].rstrip(" ,")
            for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", option)
        ]
        
        # answer = ['a', 'b', 'c', 'd', 'e'].index(correct)
        texts.append("Question: {}\na. {}\nb. {}\nc. {}\nd. {}\ne. {}\nAnswer: {}".format(problem, choices[0], choices[1], choices[2], choices[3], choices[4], correct))
    return texts

def preprocess_mgsm_val(docs):
    questions = docs["question"]
    answers = docs["answer"]
    texts = [question + "\n" + answer for question, answer in zip(questions, answers)]
    return texts


def preprocess_gsm8k_val(docs):
    instructions = docs["question"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"Question: {instruction}\nAnswer: {output}"
        texts.append(text)
    return texts


def preprocess_math(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def math_fewshot_prompt(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:" + doc["solution"]


def math_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def preprocess_finance_test(doc) -> str:
    return doc["query"]


PROCESS_FUNC = {
    # medical user
    "openlifescienceai/medmcqa": preprocess_medmcqa,
    "GBaker/MedQA-USMLE-4-options-hf": preprocess_medqa,
    "hails/mmlu_no_train": preprocess_mmlu,
    "bigbio/pubmed_qa": preprocess_pubmedqa,
    # math user
    "hails/agieval-gaokao-mathcloze": preprocess_agieval,
    "hails/agieval-gaokao-mathqa": preprocess_agieval,
    "hails/agieval-aqua-rat": preprocess_agieval,
    "hails/agieval-math": preprocess_agieval,
    "hails/agieval-sat-math": preprocess_agieval,
    "haonan-li/cmmlu": preprocess_cmmlu,
    "allenai/math_qa": preprocess_mathqa,
    "juletxara/mgsm": preprocess_mgsm,
    # "openai/gsm8k": preprocess_gsm8k,
    # math learnware
    "TIGER-Lab/MathInstruct": preprocess_alpaca_no_input_no_label,
    "meta-math/MetaMathQA": preprocess_qr_no_label,
    "meta-math/MetaMathQA-40K": preprocess_qr_no_label,
    "fxmeng/MetaMath-GSM240K": preprocess_qr_no_label,
    "meta-math/MetaMathQA_GSM8K_zh": preprocess_qr_zh_no_label,
    "meta-math/GSM8K_zh": preprocess_qa_zh_no_label,
    # "Dahoas/MATH-K-100-train": preprocess_math_k_100,
    "ScalableMath/MATH_train-cleaned_processed": preprocess_qa_no_label,
    "akjindal53244/Arithmo-Data": preprocess_qa_no_label,
    "microsoft/orca-math-word-problems-200k": preprocess_qa_no_label,
}


PROCESS_FUNC_WITH_LABEL = {
    # medical user
    "openlifescienceai/medmcqa": preprocess_medmcqa_val,
    "GBaker/MedQA-USMLE-4-options-hf": preprocess_medqa_val,
    "hails/mmlu_no_train": preprocess_mmlu_val,
    "bigbio/pubmed_qa": preprocess_pubmedqa_val,
    # math user
    "haonan-li/cmmlu": preprocess_cmmlu_val,
    "allenai/math_qa": preprocess_mathqa_val,
    "juletxara/mgsm": preprocess_mgsm_val,
    "lighteval/MATH": preprocess_math_train,
    "gsm8k": preprocess_gsm8k_val,
    # math learnware
    "TIGER-Lab/MathInstruct": preprocess_alpaca_no_input,
    "meta-math/MetaMathQA": preprocess_qr,
    "meta-math/MetaMathQA-40K": preprocess_qr,
    "fxmeng/MetaMath-GSM240K": preprocess_qr,
    "meta-math/MetaMathQA_GSM8K_zh": preprocess_qr_zh,
    "meta-math/GSM8K_zh": preprocess_qa_zh,
    # "Dahoas/MATH-K-100-train": preprocess_math_k_100,
    "ScalableMath/MATH_train-cleaned_processed": preprocess_math_train,
    "akjindal53244/Arithmo-Data": preprocess_qa,
    "microsoft/orca-math-word-problems-200k": preprocess_qa,
}


def prepare_train_data(dataset_name_str):
    if dataset_name_str in list(PROCESS_FUNC_WITH_LABEL.keys()):
        dataset = load_dataset(dataset_name_str, split="train")
        if dataset_name_str == "meta-math/GSM8K_zh": 
            dataset = dataset.filter(lambda x: x['split']=='train')
        dataset = dataset.map(lambda x: {"text": PROCESS_FUNC_WITH_LABEL[dataset_name_str](x)}, batched = True)
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    elif dataset_name_str in list(LEARNWARE_FIN.values()):
        train_dataset = load_dataset(dataset_name_str, split="train") 
        if "cra" not in dataset_name_str:
            val_dataset = load_dataset(dataset_name_str, split="valid") 
        else:
            val_dataset = load_dataset(dataset_name_str, split="validation") 
        train_dataset = train_dataset.map(lambda x: {"text": preprocess_finance(x)}, batched = True)
        val_dataset = val_dataset.map(lambda x: {"text": preprocess_finance(x)}, batched = True)
    else:
        dataset_list = dataset_name_str.split(',')
        train_datasets = []
        for dataset_name in dataset_list:
            dataset = load_dataset(dataset_name, split="train") 
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['instruction', 'input', 'output']])
            train_datasets.append(dataset)
        combined_dataset = concatenate_datasets(train_datasets)
        combined_dataset = combined_dataset.map(lambda x: {"text": preprocess_alpaca(x)}, batched = True)
        split_dataset = combined_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test'] 

    return train_dataset, val_dataset


def prepare_test_data(dataset_name_str):
    temp_list = dataset_name_str.split(",")
    subset_name = None
    if len(temp_list) != 1:
        subset_name = temp_list[1]
    dataset_name = temp_list[0]
    if subset_name:
        test_dataset = load_dataset(dataset_name, subset_name, split="test") 
    else:
        test_dataset = load_dataset(dataset_name, split="test") 
    
    if dataset_name == "gsm8k":
        rnd = random.Random(1234)
        train_dataset = load_dataset(dataset_name, "main", split="train")
        train_dataset = train_dataset.map(lambda x: {"text": preprocess_gsm8k_val(x)}, batched=True)
        train_docs = train_dataset["text"]
        fewshot_examples = rnd.sample(train_docs, 5)
        fewshot_context = (
                "\n\n".join(fewshot_examples) + "\n\n"
            )
        test_dataset = test_dataset.map(lambda x: {"text": fewshot_context + preprocess_gsm8k(x)})
    elif dataset_name == "lighteval/MATH":
        fewshot_context = (
                "\n\n".join(
                    [
                        math_fewshot_prompt(example)
                        for example in math_fewshot_samples()
                    ]
                )
                + "\n\n"
            )
        test_dataset = test_dataset.map(lambda x: {"text": fewshot_context + preprocess_math(x)})
    elif dataset_name in list(USER_FIN.values()):
        test_dataset = test_dataset.map(lambda x: {"text": preprocess_finance_test(x)})
    else:
        test_dataset = test_dataset.map(lambda x: {"text": PROCESS_FUNC[dataset_name](x)})
    return test_dataset
