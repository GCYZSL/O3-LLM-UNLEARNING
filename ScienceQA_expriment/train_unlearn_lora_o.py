import os
import sys
from typing import List

import fire
import torch
import transformers
# from peft import PeftModel
from datasets import load_dataset
from datasets import load_from_disk

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    # get_peft_model,
    # get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import Trainer

from src.mapping_hacked_o import get_peft_model
from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
from src.peft_model_hacked_o import PeftModel
from safetensors.torch import load_file as safe_load_file
import numpy as np
# from src.modeling_llama_hacked import LlamaForCausalLM
# from src.peft_model_hacked import set_peft_model_state_dict


from utils.prompter import Prompter

os.environ["WANDB_DISABLED"] = "true"

import random

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(
        # model/data params
        base_model: str = "./llama2_clsall", # "./llama2_base",  # the only required argument
        data_path: str =  "./data/clinc_text_12_intent_force/clinc_s0_train_random_force.json", #"./data/scienceqa_random_force_5/scienceqa_biology_train_random_force.json", # "./data/clinc_text_12_intent_force/clinc_s0_train_random_force.json", #"./data/scienceqa_random_k_l/scienceqa_biology_train_random_nk_l.json", # "./data/randomlabel/scienceqa/scienceqa_biology_train_random.json",# "/tank/local/cgo5577/MoLA_f/datasets/CL_biology_train_scienceq_all.json",
        output_dir: str = "./lora_bio_random_test_force_clinc",
        seed: int = 0,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 1,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        lora_weights: str = "", #"./lora_bio,./lora_bio",
        ood_weight: str = "1,1",
        orthogonal_loss: bool = True,
        orthogonal_loss_weight: float = 0.1,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = True,
        group_by_length: bool = True,  # faster, but produces an odd training loss curve
        load_in_8bit: bool = False,

        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = "", # None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    set_seed(seed)
    if isinstance(lora_weights, str):
        lora_weights = lora_weights.split(",")
        if lora_weights[0] == "":
            lora_weights = []
            print("NO LORA WEIGHTS FOR O_LOSS")
    elif isinstance(lora_weights, bool):
        lora_weights = []
        print("NO LORA WEIGHTS FOR O_LOSS")
    else:
        lora_weights = [lr for lr in lora_weights]
        if lora_weights[0] == "":
            lora_weights = []
            print("NO LORA WEIGHTS FOR O_LOSS")

    if isinstance(ood_weight, str):
        ood_weight = ood_weight.split(",")
        ood_weight = [int(lr) for lr in ood_weight]
    else:
        ood_weight = [int(lr) for lr in ood_weight]

    if isinstance(resume_from_checkpoint, bool):
        resume_from_checkpoint = False

    unlearn_lora_path = "./unlearn_lora_checkpoints"
    if not os.path.exists(unlearn_lora_path):
        os.mkdir(unlearn_lora_path)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"lora_weights: {lora_weights}\n"
            f"ood_weight: {ood_weight}\n"
            f"orthogonal_loss: {orthogonal_loss}\n"
            f"orthogonal_loss_weight: {orthogonal_loss_weight}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("load aloras")
    olora_weights = []
    for i in lora_weights:
        if len(i) == 0:
            continue
        adapters_weights = safe_load_file(os.path.join(i, "adapter_model.safetensors"))
        olora_weights.append(adapters_weights)
        print("LOAD", i, "for Onthogonal Loss" )


    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules
    # config.ood_weight = ood_weight
    if len(olora_weights) == 0:
        orthogonal_loss = False
    config.orthogonal_loss = orthogonal_loss
    config.orthogonal_loss_weight = orthogonal_loss_weight
    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
    model.init_oodweight(ood_weight=ood_weight)
    model.init_active_adapters_d(active_adapters_d=['default'])

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_from_disk(data_path)
    # else:
    #     data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        # checkpoint_name = os.path.join(
        #     resume_from_checkpoint, "pytorch_model.bin"
        # )  # Full checkpoint
        # if not os.path.exists(checkpoint_name):
        #     checkpoint_name = os.path.join(
        #         resume_from_checkpoint, "adapter_model.bin"
        #     )  # only LoRA model - LoRA config above has to fit
        #     resume_from_checkpoint = (
        #         False  # So the trainer won't try loading its state
        #     )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        # if os.path.exists(checkpoint_name):
        #     print(f"Restarting from {checkpoint_name}")
        #     adapters_weights = torch.load(checkpoint_name)
        #     set_peft_model_state_dict(model, adapters_weights)
        # else:
        #     print(f"Checkpoint {checkpoint_name} not found")
        model = PeftModel.from_pretrained(
                model,
                resume_from_checkpoint,
                is_trainable=True,
            )
        print(f"Restarting from {resume_from_checkpoint}")
    else:
        model = get_peft_model(model, config)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # fp16=True,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.train()

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
