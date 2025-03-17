import shutil
from learnware.market import instantiate_learnware_market
from learnware.specification import generate_semantic_spec
from learnware.specification.module import generate_generative_model_spec, generate_rkme_text_spec

from benchmark import Benchmark
from benchmark.config import LEARNWARE_FIN, LEARNWARE_MED, LEARNWARE_MATH

import os
import zipfile
import torch


def prepare_learnware(benchmark_name, name):
    dataset_name = name
    default_path = "learnware_pool/default/"
    
    if dataset_name == "fiqasa":
        base_model = "Meta-Llama-3.1-8B"
    elif dataset_name == "australian":
        base_model = "Meta-Llama-3.1-8B-Instruct"
    else:
        base_model = "Qwen2.5-7B"
    
    model_folder = f"models/{base_model}/{dataset_name}"
    versions = sorted(os.listdir(model_folder))
    
    for i, version in enumerate(versions):
        folder_path = f"learnware_pool/{benchmark_name}/learnwares/{dataset_name}-{i+1}"
        os.makedirs(folder_path, exist_ok=True)
        copy_adapter(folder_path, version, model_folder)
        update_from_default(folder_path, os.path.join(default_path, base_model))
        build_specification_from_cache(folder_path, dataset_name)
        zip_dir = f"learnware_pool/{benchmark_name}/zips"
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, f"{dataset_name}-{i+1}.zip")
        compress_folder_to_zip(folder_path, zip_path)


def add_learnware_to_market(benchmark_name, name, market):
    dataset_name = name
    default_path = "learnware_pool/default/"
    benchmark2scenario = {
        "medical": "Health",
        "finance": "Financial",
        "math": "Others"
    }
    
    if dataset_name == "fiqasa":
        base_model = "Meta-Llama-3.1-8B"
        base_model_path = "NousResearch/Meta-Llama-3.1-8B"
        license = "Others"
    elif dataset_name == "australian":
        base_model = "Meta-Llama-3.1-8B-Instruct"
        base_model_path = "NousResearch/Meta-Llama-3.1-8B-Instruct"
        license = "Others"
    else:
        base_model = "Qwen2.5-7B"
        base_model_path = "Qwen/Qwen2.5-7B"
        license = "Apache-2.0"
    
    model_folder = f"models/{base_model}/{dataset_name}"
    versions = sorted(os.listdir(model_folder))
    
    for i, version in enumerate(versions):
        folder_path = f"learnware_pool/{benchmark_name}/learnwares/{dataset_name}-{i+1}"
        os.makedirs(folder_path, exist_ok=True)
        copy_adapter(folder_path, version, model_folder)
        update_from_default(folder_path, os.path.join(default_path, base_model))
        build_specification_from_cache(folder_path, dataset_name)
        zip_dir = f"learnware_pool/{benchmark_name}/zips"
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, f"{dataset_name}-{i+1}.zip")
        compress_folder_to_zip(folder_path, zip_path)
    
        semantic_spec = generate_semantic_spec(
            name=f"{dataset_name}-{i+1}",
            description=f"LoRA adapter fine-tuned using SFT on the {dataset_name} dataset. Hugging Face path of its base model: {base_model_path}",
            data_type="Text",
            model_type="PEFT Model",
            task_type="Text Generation",
            library_type="PyTorch",
            scenarios=[benchmark2scenario[benchmark_name]],
            license=license,
            input_description=None,
            output_description=None,
        )
        # semantic_spec = generate_semantic_spec(
        #     name=name,
        #     description=name,
        #     data_type="Text",
        #     model_type="Base Model",
        #     task_type="Text Generation",
        #     library_type="PyTorch",
        #     scenarios=["Others"],
        #     license="Others",
        #     input_description=None,
        #     output_description=None,
        # )
        market.add_learnware(zip_path, semantic_spec)


def update_from_default(folder_path, default_path):
    for item in os.listdir(default_path):
        src_item = os.path.join(default_path, item)
        dest_item = os.path.join(folder_path, item)
        
        if not os.path.exists(dest_item):
            print(f"Copy default files to {dest_item}")
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dest_item)
            else:
                shutil.copy2(src_item, dest_item)


def copy_adapter(folder_path, version, model_folder):
    if not os.path.exists(os.path.join(folder_path, "adapter")):
        print(f"Copy adapter files from {model_folder}/{version} to {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        shutil.copytree(
            os.path.join(model_folder, version, "adapter"),
            os.path.join(folder_path, "adapter"))
            

def compress_folder_to_zip(folder_path, zip_file_path):
    """
    将指定文件夹压缩为 ZIP 文件。

    :param folder_path: 要压缩的文件夹路径
    :param zip_file_path: 生成的 ZIP 文件路径
    """
    if not os.path.exists(zip_file_path):
        print(f"Compress folder to zip_path {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 将文件添加到 ZIP 中，并保留相对路径
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)


def build_specification_from_cache(folder_path, dataset_name):
    rkme_path = os.path.join(folder_path, "rkme.json")
    generative_path = os.path.join(folder_path, "generative.pth")
    
    if not os.path.exists(rkme_path):
        print(f"Build RKME from cache to {rkme_path}")
        if dataset_name in LEARNWARE_FIN:
            src_path = f"/home/zhaozc/text_learnware/llama3-finetune/storage/rkmes/finance/reduced_set_size_100/gamma_0.1/learnware/{dataset_name}.json"
            shutil.copy2(src_path, rkme_path)
        elif dataset_name in LEARNWARE_MED:
            src_path = f"/home/zhaozc/text_learnware/llama3-finetune/storage/rkmes/medical/reduced_set_size_100/gamma_0.1/learnware/{dataset_name}.json"
            shutil.copy2(src_path, rkme_path)
        elif dataset_name in LEARNWARE_MATH:
            src_path = f"/home/zhaozc/text_learnware/llama3-finetune/storage/rkmes/math/reduced_set_size_100/gamma_0.1/learnware/{dataset_name}.json"
            shutil.copy2(src_path, rkme_path)

    if not os.path.exists(generative_path):
        print(f"Build PAVE from cache to {generative_path}")
        if dataset_name in LEARNWARE_FIN:
            finetuned_checkpoint = torch.load(f"/home/shihy/drive/LLM-finance-GridSearch-qwen/condidate-{1}/learnware-{dataset_name}/finetuned.pt", weights_only=False)
        elif dataset_name in LEARNWARE_MED:
            finetuned_checkpoint = torch.load(f"/home/shihy/drive/LLM-med-GridSearch-qwen-backup/condidate-{0}/{dataset_name}/finetuned.pt", weights_only=False)
        elif dataset_name in LEARNWARE_MATH:
            finetuned_checkpoint = torch.load(f"/home/shihy/drive/LLM-math-GridSearch-qwen/condidate-{0}/{dataset_name}/finetuned.pt", weights_only=False)
        else:
            raise NotImplementedError("Invalid dataset_name")
        
        finetuned_state_dict = finetuned_checkpoint["state_dict"]["model"]
        task_vector = torch.concatenate([
            p.reshape(-1) for n, p in finetuned_state_dict.items()
        ])
        torch.save({
            "type": "GenerativeModelSpecification",
            "task_vector": task_vector.detach().cpu()
        }, generative_path)



def build_market(benchmark_name, rebuild=True):
    llm_market = instantiate_learnware_market(market_id=f"llm_{benchmark_name}", name="llm", rebuild=rebuild)
    benchmark = Benchmark(benchmark_name)
    learnware_names = benchmark.get_learnware_names()
    print("Leanrware Names:", ", ".join(learnware_names))
    for name in learnware_names:
        title = "="*20 + name + "="*20
        print(title)
        # train_dataset, _ = benchmark.get_learnware_dataset(name)
        add_learnware_to_market(benchmark_name, name, llm_market)
        # prepare_learnware(benchmark_name, name)
        print("Market size after adding learnware:", len(llm_market))
        print("=" * len(title))


if __name__ == "__main__":
    build_market("medical")
    build_market("math")
    build_market("finance")