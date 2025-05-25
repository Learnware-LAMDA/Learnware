from typing import List
from datasets import load_dataset

DATASET = {
    "medmcqa": "openlifescienceai/medmcqa",
    "pubmedqa": "bigbio/pubmed_qa,pubmed_qa_labeled_fold0_source",
}


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


def preprocess_pubmedqa(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs,
        doc["QUESTION"],
    )


PROCESS_FUNC = {
    # medical user
    "openlifescienceai/medmcqa": preprocess_medmcqa,
    "bigbio/pubmed_qa": preprocess_pubmedqa,
}


def prepare_data(dataset_name_str):
    temp_list = dataset_name_str.split(",")
    subset_name = None
    if len(temp_list) != 1:
        subset_name = temp_list[1]
    dataset_name = temp_list[0]
    if subset_name:
        test_dataset = load_dataset(dataset_name, subset_name, split="test", trust_remote_code=True)
    else:
        test_dataset = load_dataset(dataset_name, split="test", trust_remote_code=True)
    test_dataset = test_dataset.map(lambda x: {"text": PROCESS_FUNC[dataset_name](x)})
    return test_dataset
