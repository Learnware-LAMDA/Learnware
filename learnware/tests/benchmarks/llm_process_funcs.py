import re
from typing import List


def preprocess_alpaca(docs) -> List[str]:
    alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    instructions = docs["instruction"]
    inputs = docs["input"]
    outputs = docs["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return texts


def preprocess_alpaca_no_label(docs) -> List[str]:
    alpaca_no_label_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
    instructions = docs["instruction"]
    inputs = docs["input"]
    texts = []
    for instruction, input in zip(instructions, inputs):
        text = alpaca_no_label_prompt.format(instruction, input)
        texts.append(text)
    return texts


def preprocess_alpaca_no_input(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["instruction"]
    outputs = docs["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_alpaca_no_input_no_label(docs) -> List[str]:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["instruction"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qr(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query"]
    outputs = docs["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qr_no_label(docs) -> List[str]:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["query"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qr_zh(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query_zh"]
    outputs = docs["response_zh"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qr_zh_no_label(docs) -> List[str]:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["query_zh"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qa(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qa_no_label(docs) -> List[str]:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["question"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_qa_zh(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question_zh"]
    outputs = docs["answer_zh"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_qa_zh_no_label(docs) -> List[str]:
    alpaca_no_input_no_label_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n"
    instructions = docs["question_zh"]
    texts = []
    for instruction in instructions:
        text = alpaca_no_input_no_label_prompt.format(instruction)
        texts.append(text)
    return texts


def preprocess_finance(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["query"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        instruction.rstrip(' Answer:')
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts


def preprocess_math_train(docs) -> List[str]:
    alpaca_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    instructions = docs["question"]
    outputs = docs["answer_detail"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_no_input_prompt.format(instruction, output)
        texts.append(text)
    return texts



def preprocess_medmcqa_no_label(docs) -> List[str]:
    opas = docs["opa"]
    opbs = docs["opb"]
    opcs = docs["opc"]
    opds = docs["opd"]
    questions = docs["question"]
    texts = []
    for opa, opb, opc, opd, question in zip(opas, opbs, opcs, opds, questions):
        option_choices = {
            "A": opa,
            "B": opb,
            "C": opc,
            "D": opd,
        }
        prompt = "Question: " + question + "\nChoices:\n"
        for choice, option in option_choices.items():
            prompt += f"{choice.upper()}. {option}\n"
        prompt += f"Answer:"
        texts.append(prompt)
    return texts


def preprocess_medmcqa(docs) -> List[str]:
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


def preprocess_medqa_no_label(docs) -> List[str]:
    ending0s = docs["ending0"]
    ending1s = docs["ending1"]
    ending2s = docs["ending2"]
    ending3s = docs["ending3"]
    sent1s = docs["sent1"]
    texts = []
    for sent1, ending0, ending1, ending2, ending3 in zip(sent1s, ending0s, ending1s, ending2s, ending3s):
        option_choices = {
            "A": ending0,
            "B": ending1,
            "C": ending2,
            "D": ending3,
        }
        answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
        texts.append(f"Question: {sent1}\n{answers}Answer:")
    return texts


def preprocess_medqa(docs) -> List[str]:
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


def preprocess_mmlu_no_label(docs) -> List[str]:
    questions = docs["question"]
    choices = docs["choices"]
    texts = []
    for question, options in zip(questions, choices):
        texts.append(
            "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:".format(
                question.strip(),
                options[0],
                options[1],
                options[2],
                options[3]
            )
        )
    return texts


def preprocess_mmlu(docs) -> List[str]:
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


def preprocess_pubmedqa_no_label(docs) -> List[str]:
    contexts_list = docs["CONTEXTS"]
    questions = docs["QUESTION"]
    texts = []
    for contexts, question in zip(contexts_list, questions):
        ctxs = "\n".join(contexts)
        texts.append("Abstract: {}\nQuestion: {}\nAnswer:".format(ctxs, question))
    return texts


def preprocess_pubmedqa(docs) -> List[str]:
    contexts_list = docs["CONTEXTS"]
    questions = docs["QUESTION"]
    answers = docs["final_decision"]
    texts = []
    for contexts, question, answer in zip(contexts_list, questions, answers):
        ctxs = "\n".join(contexts)
        texts.append("Abstract: {}\nQuestion: {}\nAnswer: {}".format(ctxs, question, answer))
    return texts


def preprocess_agieval_no_label(docs) -> List[str]:
    return docs["query"]


def preprocess_cmmlu_no_label(docs) -> List[str]:
    questions = docs["Question"]
    as_ = docs["A"]
    bs = docs["B"]
    cs = docs["C"]
    ds = docs["D"]
    texts = []
    for question, a, b, c, d in zip(questions, as_, bs, cs, ds):
        texts.append("{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：".format(
            question.strip(), a, b, c, d
        ))
    return texts


def preprocess_cmmlu(docs) -> List[str]:
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


def preprocess_mathqa_no_label(docs) -> List[str]:
    problems = docs["Problem"]
    texts = [f"Question: {problem}\nAnswer:" for problem in problems]
    return texts


def preprocess_mathqa(docs) -> List[str]:
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


def preprocess_mgsm_no_label(docs) -> List[str]:
    questions = docs["question"]
    texts = [f"问题: "+question+"\n逐步解答:" for question in questions]
    return texts


def preprocess_mgsm(docs) -> List[str]:
    questions = docs["question"]
    answers = docs["answer"]
    texts = [question + "\n" + answer for question, answer in zip(questions, answers)]
    return texts


def preprocess_gsm8k_no_label(docs) -> List[str]:
    questions = docs["question"]
    texts = [f"Question: {question}\nAnswer:" for question in questions]
    return texts


def preprocess_gsm8k(docs) -> List[str]:
    instructions = docs["question"]
    outputs = docs["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"Question: {instruction}\nAnswer: {output}"
        texts.append(text)
    return texts


def preprocess_math_no_label(docs) -> List[str]:
    problems = docs["problem"]
    texts = ["Problem:" + "\n" + problem + "\n\n" + "Solution:" for problem in problems]
    return texts


def preprocess_finance_no_label(docs) -> List[str]:
    return docs["query"]

