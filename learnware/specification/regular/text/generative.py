from __future__ import annotations

import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import trl
import torch

from torch import nn

from trl import SFTConfig
from peft import LoraConfig, PeftModel
from datasets import Dataset

from transformers import (
    PreTrainedModel,
    TrainingArguments,
    Qwen2ForCausalLM,
    Qwen2Tokenizer
    )

from peft import get_peft_model

from ..base import TaskVectorSpecification
from ....logger import get_module_logger
from ....utils import allocate_cuda_idx, choose_device

logger = get_module_logger("GenerativeModelSpecification", "INFO")


class GenerativeModelSpecification(TaskVectorSpecification):
    """Task Vector Specification for Large Language Model"""

    def __init__(self,
                 cuda_idx: int = None,
                 attn_implementation: str = "eager",
                 per_device_train_batch_size: int = 2,
                 gradient_accumulation_steps: int = 1,
                 max_seq_length: int = 2048,
                 **kwargs):
        """Initializing Task Vector Specification's parameters.
        
        Parameters
        ----------
        cuda_idx : int, optional
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device
        
        attn_implementation : str, optional
            The type of attention implementation to use. Default is 'eager'.

        per_device_train_batch_size : int, optional
            The training batch size for each device. Default is 2.

        gradient_accumulation_steps : int, optional
            The number of steps to accumulate gradients before an optimizer step.
            Default is 1.

        max_seq_length : int, optional
            The maximum sequence length for the model input. Default is 2048.

        **kwargs : dict
            Additional keyword arguments.
        """
        super(GenerativeModelSpecification, self).__init__(type=self.__class__.__name__)
        
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self._device = choose_device(cuda_idx=self._cuda_idx)
        
        self._task_vector = None
        
        self.attn_implementation = attn_implementation
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        
        self.__extra_args = {
            "weight_decay_l1": 1.0,
            "weight_decay_l2": 0.5,
            "max_steps": 400,
            "lr": 1e-5,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.03,
        }
        
        
    @property
    def task_vector(self):
        if self._task_vector is None:
            raise Exception("Call generate_stat_spec_from_data first!")
        
        return self._task_vector
    
    @task_vector.setter
    def task_vector(self, value):
        self._task_vector = value
        
    def generate_stat_spec_from_data(
        self,
        dataset: Optional[Dataset] = None,
        dataset_text_field="text",
        X: List[str] = None,
        verbose: bool = True,
        beimingwu = True,
        **kwargs
    ):
        """Initializing Task Vector Specification's parameters.
        
        Parameters
        ----------
        
        dataset_text_field : str, optional
            Name of the text field of the dataset. Default is "text".
        
        """
        if dataset is None:
            assert X is not None, "X and dataset cannot both be None."
            dataset = Dataset.from_dict({dataset_text_field: X})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer, model = self._init_tokenizer_model(beimingwu)
            trainer_config = self._trainer_config(temp_dir, dataset_text_field)
            trainer = self._init_trainer(model, tokenizer, dataset, trainer_config)
                
            param_0 = [p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad]
            trainer.train()
            param_1 = [p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad]

        self._task_vector = torch.concatenate([
            (p1 - p0).reshape(-1) for p0, p1 in zip(param_0, param_1)
        ])
    
    
    def _init_tokenizer_model(self, beimingwu):
        """
        Initialize foundational model (e.g. Qwen) used for task vector generation.
        And, this method should not be overridden if the specification needs to be submitted to Beimingwu.
        """
        if beimingwu:
            from ....client import LearnwareClient

            client = LearnwareClient()
            base_model_path = client.get_pretrained_path("00002890")
        else:
            base_model_path = "Qwen/Qwen2.5-0.5B"
        
        set_seed(3407)    
        tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)
        model = Qwen2ForCausalLM.from_pretrained(
            base_model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
        ).to(self._device)
        
        if beimingwu:
            client = LearnwareClient()
            adapter_path = client.get_pretrained_path("00002891")
            model = PeftModel.from_pretrained(model, adapter_path)
            
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.requires_grad = True
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj"]
            )        
            model = get_peft_model(model, peft_config)
            
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad = False
            
        return tokenizer, model

        
    def _init_trainer(self, model, tokenizer, train_dataset, args):
        
        # TODO: set_seed(3407)
        trainer = CustomSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            weight_decay_l1=self.__extra_args["weight_decay_l1"],
            args=args,
        )
        # Work around trl package bug with multi-GPU parallelism
        trainer.args._n_gpu = 1
        
        return trainer
    

    def _trainer_config(self, temp_dir, dataset_text_field):
        training_params = SFTConfig(
            output_dir=temp_dir,  # 结果路径
            max_steps=self.__extra_args["max_steps"],
            per_device_train_batch_size=self.per_device_train_batch_size,  # 这是每个GPU的训练批次大小
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # 累积多个步骤的梯度，以有效地增加批次大小
            learning_rate=self.__extra_args["lr"],  # 初始学习率
            weight_decay=self.__extra_args["weight_decay_l2"],  # 权重衰减率
            optim="adamw_torch",  # 优化器
            eval_strategy="no",
            save_strategy="no",
            # fp16=True,  # 启用混合精度训练
            # bf16=True,  # 启用BF16
            max_grad_norm=self.__extra_args["max_grad_norm"],  # 裁剪梯度
            warmup_ratio=self.__extra_args["warmup_ratio"],  # 训练开始时的预热样本比例
            group_by_length=True,  # 将训练数据集中大致相同长度的样本分组到同一batch中，提升prefill效率
            lr_scheduler_type="cosine",  # 学习率调度器衰减策略
            ddp_timeout=180000000,
            dataset_text_field=dataset_text_field,
            max_seq_length=self.max_seq_length,
            dataloader_num_workers=16,
            seed = 3407,
        )
        
        return training_params
    
    
    def save(self, filepath: str):
        torch.save({
            "type": self.type,
            "task_vector": self.task_vector.detach().cpu()
        }, filepath)
    
    
    def load(self, filepath: str):
        state = torch.load(filepath, weights_only=True)
        if state["type"] != self.type:
            logger.warning("{} may not be consistent with this class {}.".format(
                state["type"], self.type
            ))
        self._task_vector = state["task_vector"].to(self._device)
    
    
class CustomSFTTrainer(trl.SFTTrainer):
    
    def __init__(self, weight_decay_l1=None, **kwargs):
        super().__init__(**kwargs)       
        model: Union[PreTrainedModel, nn.Module] = kwargs["model"]
        args: TrainingArguments = kwargs["args"]
        
        if hasattr(args, "weight_decay_l1") and (weight_decay_l1 is not None):
            print("Warning! weight_decay_l1 is overwrited by key args.")
        if weight_decay_l1 is not None:
            self.weight_decay_l1 = weight_decay_l1
        elif hasattr(args, "weight_decay_l1"):
            self.weight_decay_l1 = args.weight_decay_l1
        else:
            assert False, "weight_decay_l1 shounld be given."
        
        self.parameters_l1_regularized = None
        
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        self.parameters_l1_regularized = [
            (p, torch.nn.Parameter(p.clone().detach())) for n, p in self.model.named_parameters() if p.requires_grad
        ]
        
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial,
                             ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # implement custom logic here
        default_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        if self.weight_decay_l1 > 0:
            l1_norm = sum((torch.linalg.norm(p - p0, 1) for p, p0 in self.parameters_l1_regularized))
            # We mask lora_A after init.
            l1_norm = self.weight_decay_l1 / len(self.parameters_l1_regularized) * l1_norm
            loss = default_loss + l1_norm
        else:
            loss = default_loss
        
        return (loss, outputs) if return_outputs else loss
    

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True