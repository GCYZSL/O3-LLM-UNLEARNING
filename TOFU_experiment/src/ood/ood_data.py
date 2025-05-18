import sys
print(sys.path)
sys.path.append("src")
import random

from dataset import ToFU
from dataset.Base import UnlearnDataset
from torch.utils.data import ConcatDataset


def load(dataset_names, tokenizer, dataset_seed, is_id):
    if is_id:
        dataset = ToFU("TOFU", subset=dataset_names)
        full_set = tofu_tokenization(dataset.dataset["test"], tokenizer, dataset_names, if_paraphase=True)
        
        forget_ratio = 0.98
        # print(len(full_set))
        length = forget_ratio * len(full_set)
        
        forget_index_list = random.sample(range(len(full_set)), int(length))
        retain_index_list = list(set(range(len(full_set))) - set(forget_index_list))

        train_set = [full_set[i] for i in forget_index_list]
        test_set = [full_set[i] for i in retain_index_list]

        return train_set, test_set
    
    else:
        dataset = ToFU("TOFU", subset=dataset_names)
        full_set = tofu_tokenization(dataset.dataset["test"], tokenizer, dataset_names)
        train_set = None
        test_set = full_set

        return train_set, test_set


def tofu_tokenization(full_set, tokenizer, dataset_name, if_paraphase=False, max_seq_length=256):
        def preprocess_train(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
            }

            prompt_1 = examples["question"]
            
            tokenized_1 = tokenizer(
                prompt_1,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
            )
            results["input_ids"].append(tokenized_1.input_ids)
            results["attention_mask"].append(tokenized_1.attention_mask)


            if if_paraphase:
                prompt_2 = examples["paraphrased_question"]
                tokenized_2 = tokenizer(
                    prompt_2,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=True,
                )
                results["input_ids"].append(tokenized_2.input_ids)
                results["attention_mask"].append(tokenized_2.attention_mask)
                
            return results
        if "retain" in dataset_name:
            train_dataset = full_set.map(preprocess_train, remove_columns=["question", "answer", 'paraphrased_answer', 'perturbed_answer', 'paraphrased_question'])
        else:
            train_dataset = full_set.map(preprocess_train, remove_columns=["question", "answer", 'perturbed_answer'])
        res = []
        for i in range(len(train_dataset["input_ids"])):
            res_t = {"input_ids": train_dataset["input_ids"][i][0], "attention_mask": train_dataset["attention_mask"][i][0]}
            res.append(res_t)
        # print(res)

        return res
