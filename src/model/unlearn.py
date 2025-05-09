import os
import sys

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RobertaTokenizer


import torch
sys.path.append("src")
from dataset import get_dataset
from metrics import eval_tofu
from optim import create_sophia_optimizer
from unlearn import get_unlearn_method
# from peft import get_peft_model, LoraConfig
from peft import LoraConfig
import pickle
import json

from model_src.mapping_hacked_o import get_peft_model
from model_src.modeling_llama_hacked_o import LlamaForCausalLM_ood
from model_src.peft_model_hacked_o import PeftModel
from safetensors.torch import load_file as safe_load_file
from model_src.mapping_hacked_o import get_peft_model
from model_src.ood_model_selector import RobertaForSelector_inference
from ood.run_ood import obtain_weights
from ood.ood_data import load

class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.alpha = kwargs.get("alpha", None)
        self.gamma = kwargs.get("gamma", None)
        self.task_name = kwargs.get("task_name", None)
        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.forget_epoch = kwargs.get("forget_epoch", 1)
        self.if_llama = "llama" in self.model_name
        self.use_lora = kwargs.get("use_lora", False)
        self.resume_path = kwargs.get("resume_path", None)

        ### modified
        self.lora_target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ]

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=self.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.lora_weights = []
    def init_model(self):
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.bfloat16,
        #     cache_dir=self.cache_dir,
        #     low_cpu_mem_usage=True,
        #     device_map="auto",
        # )

        ### modified
        print("load aloras")
        olora_weights = []
        for i in self.lora_weights:
            if len(i) == 0:
                continue
            adapters_weights = safe_load_file(os.path.join(i, "adapter_model.safetensors"))
            olora_weights.append(adapters_weights)
            print("LOAD", i, "for Onthogonal Loss")

        base_config = AutoConfig.from_pretrained(self.model_name)
        base_config.lora_target_modules = self.lora_target_modules

        orthogonal_loss = True
        if len(olora_weights) == 0:
            orthogonal_loss = False
        base_config.orthogonal_loss = orthogonal_loss

        base_config.orthogonal_loss_weight = 0.1
        model = LlamaForCausalLM_ood.from_pretrained(
            self.model_name,
            config=base_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
        ood_weight = [1,1]
        model.init_oodweight(ood_weight=ood_weight)
        model.init_active_adapters_d(active_adapters_d=['default'])

        # if self.use_lora:
        #     peft_config = LoraConfig(
        #         r=8,
        #         lora_alpha=32,
        #         target_modules=["q_proj","v_proj"],
        #         lora_dropout=0.05,
        #         bias="none",
        #         task_type="CAUSAL_LM"
        #     )
        #     model = get_peft_model(model, peft_config)
        #     print(model.print_trainable_parameters())

        if len(self.lora_weights) > 0:
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights[-1],
                is_trainable=True,
            )
            print(f"Restarting from {self.lora_weights}")
        else:
            model = get_peft_model(model, self.lora_config)
        model.train()
        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )
        self.steps_per_epoch = len(unlearn_dataset) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin" or self.unlearn_method == "sys":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=False,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            save_steps=self.max_steps,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            save_total_limit=1,
        )
        if self.optimizer is not None:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                forget_epoch=self.forget_epoch,
                optimizers=(self.optimizer, None),
            )
        else:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                forget_epoch=self.forget_epoch,
            )

    def init_optimizer(self):
        if self.sophia:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=self.lr,
                betas=self.betas,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = None

    def eval(self, logger, initial_if, ood_weights=None):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.unlearn_method == "sys":
            if_system = True
        else:
            if_system = False
        if self.resume_path is not None:
            model_name = self.resume_path
        elif initial_if:
            # model_name = os.path.join(root, "checkpoints")
            print('*'*30 + 'load initial model' + '*'*30)
            model_name = self.model_name
        else:
            model_name = os.path.join(root, "checkpoints")
            print('*'*30 + 'load unlearned model' + '*'*30)


        ### modified
        base_config = AutoConfig.from_pretrained(self.model_name)
        base_config.lora_target_modules = self.lora_target_modules

        orthogonal_loss = False
        base_config.orthogonal_loss = orthogonal_loss

        base_config.orthogonal_loss_weight = 0.1
        model = LlamaForCausalLM_ood.from_pretrained(
            self.model_name,
            config=base_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=[])
        ood_weight = [1, 1]
        model.init_oodweight(ood_weight=ood_weight)
        model.init_active_adapters_d(active_adapters_d=['default'])

        if len(self.lora_weights) > 0:
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights[-1],
                is_trainable=True,
            )
            print(f"Restarting from {self.lora_weights}")
        else:
            model = get_peft_model(model, self.lora_config)

        model.eval()

        if self.task_name != 'tofu':
            eval_ppl(model_name=model_name, output_dir=root)
            eval_few_shots(model_name=model_name, output_dir=root)
        torch.cuda.empty_cache()
        if self.task_name == "toxic":
            eval_toxic(
                model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
            )
        elif self.task_name == "copyright":
            eval_copyright(model_name=model_name, output_dir=root,batch_size=16,if_llama=self.if_llama)
        elif self.task_name == "tofu":
            forget_subset = self.dataset_names["forget"].split("_")[1]
            retain_subset = self.dataset_names["retain"].split("_")[1]

            eval_tofu(
                model=model,
                model_name=model_name,
                output_dir=root,
                forget_subset=forget_subset,
                retain_subset=retain_subset,
                if_llama=self.if_llama,
                if_system=if_system,
                ood_weights=ood_weights
            )

    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def get_ood_weights(self, forget_name, retain_name, ood_tokenizer, ood_models, ood_clrs, ood_gmm_w_cls, ood_x0, ood_mean_lists, ood_precision_lists, ood_fea_lists, task_id):
        test_names = [forget_name, retain_name, 'real_authors', 'world_facts']
        ood_weights_dict = {set_t: [] for set_t in test_names}
        for test_name in test_names:
            _, test_set = load(test_name, ood_tokenizer, dataset_seed=1000, is_id=False)
            
            for s_t in test_set:
                max_ood = 0
                for i in range(2, 3):
                    mah_score = ood_models[i].get_unsup_Mah_score_s(s_t, ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i], self.device)[:, 1:]
                    test_score = ood_clrs[i].decision_function(mah_score)
                    w_ood = obtain_weights(test_score, ood_gmm_w_cls[i], None, ood_x0[i])
                    if w_ood > max_ood:
                        max_ood = w_ood
                # print(max_ood)
                ood_weights_dict[test_name].append(max_ood)
            print(f"OOD weights for {test_name}:")
            print(sum(ood_weights_dict[test_name]) / len(ood_weights_dict[test_name]))
        
        return ood_weights_dict

    def run(self, logger):
        self.init_model()
        # evaluate 
        ood_weights = []
        for i in ["forget01", "forget05", "forget10"]:
            o_p = "./ood_checkpoints/" + f"{i}"
            ood_weights.append(o_p)
        ood_type = "ocsvm"
        ood_base_model = "roberta-large"
        ood_tokenizer = RobertaTokenizer.from_pretrained(ood_base_model)
        ood_models = []
        ood_clrs = []
        ood_thresholds = []
        ood_x0 = []
        ood_mean_lists = []
        ood_precision_lists = []
        ood_fea_lists = []
        ood_gmm_w_cls = []

        for i in ood_weights:
            roberta_path = i + f"_roberta"
            ocsvm_path = i + f"_{ood_type}.pkl"
            threshold_path = i + f"_{ood_type}_threshold.json"
            mean_list_path = i + f"_mean_list.pt"
            precision_list_path = i + f"_precision_list.pt"
            fea_list_path = i + f"_fea_list.pt"
            gmm_w_path = i + f"_gmm_w.pkl"

            ood_models.append(RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(self.device))
            with open(ocsvm_path, "rb") as input_file:
                c_lr = pickle.load(input_file)
            ood_clrs.append(c_lr)
            with open(gmm_w_path , "rb") as input_file:
                gmm_w = pickle.load(input_file)
            ood_gmm_w_cls.append(gmm_w)
            with open(threshold_path) as f:
                threshold = json.load(f)
            ood_thresholds.append(threshold[0])
            ood_x0.append(threshold[1])

            ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(self.device)))
            ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(self.device)))
            ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(self.device)))
        
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")

        ### modified
        root = logger.get_root()
        self.lora_weights.append(os.path.join(root, "checkpoints"))
        task_id = 1
        cur_ood_weights_dict = self.get_ood_weights('forget01', 'retain99', ood_tokenizer, ood_models, ood_clrs, ood_gmm_w_cls, ood_x0, ood_mean_lists, ood_precision_lists, ood_fea_lists, task_id)
        self.eval(logger, False, cur_ood_weights_dict)

        # following tasks
        forget_list = ['Tofu_forget05', 'Tofu_forget10']
        retain_list = ['Tofu_retain95', 'Tofu_retain90']

        for task_i in range(2):
            self.dataset_names["forget"] = forget_list[task_i]
            self.dataset_names["retain"] = retain_list[task_i]
            # self.eval(logger, True)
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()

            ### modified
            index = str(task_i)
            s_index = str(task_i + 1)
            logger.ckpt_root = logger.ckpt_root.replace(f"stage_{index}", f"stage_{s_index}")
            self.save(logger)
            self.lora_weights.append(os.path.join(root, "checkpoints").replace(f"stage_{index}", f"stage_{s_index}"))

            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            task_id += 1
            cur_ood_weights_dict = self.get_ood_weights(forget_list[task_i].split("_")[1], retain_list[task_i].split("_")[1], ood_tokenizer, ood_models, ood_clrs, ood_gmm_w_cls, ood_x0, ood_mean_lists, ood_precision_lists, ood_fea_lists, task_id)
            self.eval(logger, False, cur_ood_weights_dict)


def get(**kwargs):
    return Unlearn(**kwargs)
