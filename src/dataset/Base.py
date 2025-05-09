import random

import torch
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        self.dataset_name = dataset_name
        self.with_normal = with_retain
        self.if_llama = if_llama
        self.question_start_token = "[INST] " if self.if_llama else "### Question: "
        self.question_end_token = " [\INST]" if if_llama else "\n"
        self.answer_start_token = " " if if_llama else "### Answer: "

    def get_dataset(self):
        pass

    def __preprocess__(self, tokenizer, forget_ratio, dataset_seed):
        pass

    def build_dataset(self, tokenizer, forget_ratio, dataset_seed):
        pass


class UnlearnDataset(Dataset):
    def __init__(self, datasets, forget_ratio, dataset_seed, self_retain=False, subset=None):
        self.forget_ratio = forget_ratio
        self.dataset_seed = dataset_seed
        self.self_retain = self_retain
        self.forget_index_list = None
        self.retain_index_list = None

        if "forget" in datasets.keys():
            self.forget_dataset = datasets["forget"]
        else:
            self.forget_dataset = None

        if "retain" in datasets.keys():
            self.retain_dataset = datasets["retain"]
        else:
            self.retain_dataset = None

        self.build_unlearn_dataset(subset)

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        elif self.retain_dataset:
            return len(self.retain_dataset)
        else:
            raise ValueError("No dataset")

    def build_unlearn_dataset(self, subset):
        if self.forget_dataset:
            if self.forget_ratio > 1:
                length = int(self.forget_ratio)

            elif self.forget_ratio <= 1 and self.forget_ratio > 0:
                length = int(len(self.forget_dataset) * self.forget_ratio)

            random.seed(self.dataset_seed)
            forget_index_list = random.sample(range(len(self.forget_dataset)), length)
            self.forget_index_list = forget_index_list
            if self.self_retain:
                retain_index_list = list(
                    set(range(len(self.forget_dataset))) - set(self.forget_index_list)
                )
                self.retain_dataset = self.forget_dataset.select(retain_index_list)
                self.retain_index_list = retain_index_list
            if subset != None:
                if subset == 'Tofu_forget05' or subset == 'forget05':
                    print('hit 05')
                    self.forget_index_list = [x for x in forget_index_list if x < 160]
                elif subset == 'Tofu_forget10' or subset == 'forget10':
                    print('hit 10')
                    self.forget_index_list = [x for x in forget_index_list if x < 200]
            self.forget_dataset = self.forget_dataset.select(self.forget_index_list)

    def __getitem__(self, idx):
        if self.forget_dataset:
            forget_data = self.forget_dataset[idx]
            if self.retain_dataset:
                retain_idx = random.randint(0, len(self.retain_dataset) - 1)
                retain_data = self.retain_dataset[retain_idx]
                return {"forget": forget_data, "retain": retain_data}
            else:
                return {"forget": forget_data, "retain": None}
        else:
            retain_data = self.retain_dataset[idx]
            return {"forget": None, "retain": retain_data}


def unlearncollector(samples):
    res = {}
    if samples[0]["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples]),
            torch.stack([sample["refused_label"] for sample in forget_samples]),
            torch.stack([sample["question_length"] for sample in forget_samples]),
        )
    else:
        res["forget"] = None
    if samples[0]["retain"]:
        retain_samples = [sample["retain"] for sample in samples]
        res["retain"] = (
            torch.stack([sample["input_ids"] for sample in retain_samples]),
            torch.stack([sample["attention_mask"] for sample in retain_samples]),
            torch.stack([sample["label"] for sample in retain_samples]),
        )
    else:
        res["retain"] = None
    return res
