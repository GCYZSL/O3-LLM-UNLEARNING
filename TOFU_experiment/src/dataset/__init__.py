from collections import defaultdict

from transformers import default_data_collator

from .Base import UnlearnDataset, unlearncollector
from .Tofu import ToFU

def get_dataset(
    dataset_names,
    tokenizer,
    dataset_seed,
    forget_ratio,
    self_retain=False,
    if_llama=False,
):
    ### forget dataset & test dataset

    if "Tofu" in dataset_names["forget"]:
        subset = dataset_names["forget"].split("_")[1]
        dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif "forget" not in dataset_names:
        forget_dataset = None
        test_dataset = None
    else:
        raise ValueError("No dataset")

    #### retain dataset
    if "Tofu" in dataset_names["retain"]:
        subset = dataset_names["retain"].split("_")[1]
        dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        retain_dataset = dataset["train"]
    elif "retain" not in dataset_names:
        retain_dataset = None
    else:
        raise ValueError("No dataset")

    unlearn_dataset = UnlearnDataset(
        {"forget": forget_dataset, "retain": retain_dataset},
        forget_ratio,
        dataset_seed,
        self_retain,
        dataset_names["forget"],
    )
    unlearn_collator = unlearncollector

    test_collator = default_data_collator

    return unlearn_dataset, test_dataset, unlearn_collator, test_collator


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_names = {"forget": "SafePku", "retain": "BookCorpus"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False
    unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )
    print(len(unlearn_dataset))

    print(len(test_dataset))
    import torch

    dataloader = torch.utils.data.DataLoader(
        unlearn_dataset, batch_size=2, collate_fn=unlearn_collator
    )
    for batch in dataloader:
        print(batch)
        break
