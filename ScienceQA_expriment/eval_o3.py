import os
import re
import json
import argparse
import torch
import random
from tqdm import tqdm

from src.ood_model_selector import RobertaForSelector, RobertaForSelector_inference

from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer

from src.peft_model_hacked_o import PeftModel
import pickle
from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
import math
import sys
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig, AutoTokenizer
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

import json
import os.path as osp
from typing import Union

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

def knowledge_weights(gmm_scores, threshold_train):
    weight_res = []
    r = 3  ## this value can be changed to adjust the range of id data
    gmm_scores -= r * threshold_train

    for i in range(gmm_scores.shape[0]):
        weight_t = math.exp(gmm_scores[i]) / (1 + math.exp(gmm_scores[i]))
        weight_res.append(weight_t)

    return weight_res

def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    # gmm.covariances_ = np.array([[[std1**2]], [[std2**2]]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    # center: x0
    # x0 = minimize(lambda x: (gmm_cdf(x, gmm) - 0.5)**2, x0=0).x[0]
    x0 = (mean1 + mean2) / 2

    return gmm, x0

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# 6. the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

all_w_res = []
all_w_res_dic = {}
def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2 # 2 0.5

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    # if w_res > 0.4:
    #     w_res = 1.2
    # elif w_res <= 0.4 and w_res > 0.3:
    #     w_res = w_res * 1.2 # 2
    # else:
    #     w_res=0

    if w_res > 0.9: # not_test_RD 0.997 0.995 0.98 # test_SD 0.999 0.990 0.9845
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res # 2
    else:
        w_res=0

    return w_res
# print(np.mean(all_w_res), min(all_w_res), max(all_w_res))
# mean, min, max 0.3775407149001829 0.37754066879814546 0.37755057892780686 scienceqa_not_biology_physics_chemistry_economics_test_RD
# 0.4683295632561949 0.37754066879814546 0.985614423543908  1192/1581 0.377 scienceqa_biology_test_SD
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # if not template_name:
        #     # Enforce the default here, so the constructor can be called with '' and will not break.
        #     template_name = "alpaca"
        # file_name = osp.join("templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        # with open(file_name) as fp:
        #     self.template = json.load(fp)
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }

        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    # Defining arguments
    # ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json
    # ./data/scienceqa_RD_5/scienceqa_not_biology_test_RD.json",
    # ./data/scienceqa_standard_5/scienceqa_biology_train_std.json
    # ./data/scienceqa_SD_5/scienceqa_biology_physics_test_SD.json

    parser.add_argument('--test_dataset', type=str, default="./data/scienceqa_SD_5/scienceqa_biology_test_SD.json",
                        help='test_dataset')
    parser.add_argument('--base_model', type=str, default="gcyzsl/O3_LLAMA2_ScienceQA", help='base_model')
    parser.add_argument('--ood_base_model', type=str, default="roberta-large", help='base_model')
    parser.add_argument('--lora_weights', type=str, default= "./SCALE_0.1_seed_1_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force",
                        help='lora_model')
    # ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force
    parser.add_argument('--ood_weights', type=str, default= "./ood_checkpoints_scienceqa_1/",#"./ood_checkpoints_new_0/",  # o: 6.05 2.02
                        help='ood_model')
    parser.add_argument('--ood_type', type=str, default="biology",  # o: 6.05 2.02
                        help='ood type')
    # biology
    parser.add_argument('--ood_setting', type=str, default="c",  # o: 6.05 2.02
                        help='ood setting')
    parser.add_argument('--ood_setting_name', type=str, default="scienceqa",  # o: 6.05 2.02
                        help='ood setting name')
    parser.add_argument('--seed', type=int, default=1, help='base_model')

    # SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force
    # ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json 6.3% 1.1718236274499052 0.4530488025577745 1.2

    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    base_model = args.base_model
    max_batch_size = 1
    lora_weight = args.lora_weights
    types = args.ood_type.split("_")
    ood_types = []
    for i in types:
        if len(i) > 0:
            ood_types.append(i)
    ood_setting = args.ood_setting
    ood_setting_names = args.ood_setting_name
    print(args.test_dataset)
    print(args.base_model)
    print(args.lora_weights)
    print(args.ood_weights)
    print(ood_types)
    print(ood_setting)
    print(args.ood_setting_name)

    ood_weights = [] # ["biology", "physics", "chemistry", "economics", 'earth-science']
    for i in ood_types:  # "biology", "physics", "chemistry", "economics"
        o_p = args.ood_weights + f"{ood_setting_names}_{i}_ood_{ood_setting_names}"
        ood_weights.append(o_p)
    ood_type = "ocsvm"

    path = "/".join(lora_weight.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    result_file = path + "/test_noretain_{}_seed{}_oodlora_{}_{}".format(ood_setting,str(args.seed), lora_weight.split("/")[-1],
                                                args.test_dataset.split("/")[-1])
    print(result_file)


    load_8bit = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
    lora_target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ]
    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules
    # config.ood_weight = ood_weight
    orthogonal_loss = False
    olora_weights = {}
    config.orthogonal_loss = orthogonal_loss
    config.orthogonal_loss_weight = 0.1
    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weight,
        torch_dtype=torch.bfloat16,
    )
    model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
    model.init_active_adapters_d(active_adapters_d=['default'])

    print(model.config.pad_token_id, tokenizer.pad_token_id)
    print(model.config.bos_token_id, tokenizer.bos_token_id)
    print(model.config.eos_token_id, tokenizer.eos_token_id)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    ood_base_model = args.ood_base_model
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
        roberta_path = i + f"_roberta_{ood_type}"
        ocsvm_path = i + f"_{ood_type}.pkl"
        threshold_path = i + f"_threshold_{ood_type}.json"
        mean_list_path = i + f"_mean_list_{ood_type}.pt"
        precision_list_path = i + f"_precision_list_{ood_type}.pt"
        fea_list_path = i + f"_fea_list_{ood_type}.pt"
        gmm_w_path = i + f"_gmm_w_{ood_type}.pkl"


        ood_models.append(RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(device))
        with open(ocsvm_path, "rb") as input_file:
            c_lr = pickle.load(input_file)
        ood_clrs.append(c_lr)
        with open(gmm_w_path , "rb") as input_file:
            gmm_w = pickle.load(input_file)
        ood_gmm_w_cls.append(gmm_w)
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_thresholds.append(threshold[1])
        ood_x0.append(threshold[0])

        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))

    prompter = Prompter(template_name="alpaca")
    max_new_tokens = 128
    save_every = 200

    correct = 0
    results = []
    outputs = []
    gt = []
    options = ["A", "B", "C", "D", "E"]

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]
        answers = [str(example["answer"]) for example in batch]
        # print(answers)
        # generate prompt
        prompts = [prompter.generate_prompt(example['instruction'], example['input']) for example in batch]
        # print(prompts)

        ood_input = ood_tokenizer(prompts, padding='max_length', truncation=True, max_length=512, return_tensors="pt") # 256
        max_ood = 0

        for i in range((len(ood_weights))):
            mah_score = ood_models[i].get_unsup_Mah_score_s(ood_input, ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i])[:, 1:]
            test_score = ood_clrs[i].score_samples(mah_score)
            w_ood = obtain_weights(test_score, ood_gmm_w_cls[i], ood_x0[i])
            if w_ood > max_ood:
                max_ood = w_ood

        all_w_res.append(max_ood)
        dic_key = str(max_ood)[:5]
        if dic_key in all_w_res_dic:
            all_w_res_dic[dic_key] += 1
        else:
            all_w_res_dic[dic_key] = 1

        print("ood_weight: ", [1, max_ood])
        model.init_oodweight(ood_weight=[1, max_ood])


        # generate prediction
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                # generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s)
        output = [prompter.get_response(otp) for otp in output]

        # extract the answer
        # print(output)
        # pattern = re.compile(r'The anwser to the question is (\d+):*')
        pattern = re.compile(r'The answer is ([A-Z]).')
        res = [pattern.findall(otp) for otp in output]
        # print(res)
        pred = []
        for r_i in range(len(res)):
            if len(res[r_i]) == 1:
                answer = res[r_i][0]  # 'A', 'B', ...
            else:
                answer = "FAILED"
            #     print("*******************************************", res[r_i])
            pred.append(answer)
            results.append(res[r_i])
            outputs.append(output[r_i])
            gt.append(answers[r_i])

            if str(answer) == str(answers[r_i]):
                correct += 1
                print('correct:', str(answer), str(answers[r_i]))
            else:
                print('gt-ans:', str(answer), str(answers[r_i]))

        acc = correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            print(f"{len(results)}/{len(data_a)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            data = {}
            data['acc'] = acc
            data['correct'] = correct
            data['len'] = len(results)
            data['results'] = results
            data['outputs'] = outputs
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))
    print(np.mean(all_w_res), min(all_w_res), max(all_w_res))



if __name__ == "__main__":
    main()