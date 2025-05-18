import json
from datasets import load_from_disk, load_dataset
import datasets
import pandas as pd
from typing import Union
from tqdm import tqdm

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
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

def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution

def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return input, output, text

bio_topics = ['biology']
bio_phy_topics = ['biology', 'physics']
bio_phy_che_topics = ['biology', 'physics', 'chemistry']
bio_phy_che_eco_topics = ['biology', 'physics', 'chemistry', 'economics']
bio_phy_che_eco_ear_topics = ['biology', 'physics', 'chemistry', 'economics', 'earth-science']

splits = ['test', 'validation', 'train']

subject = {}
topic = {}

options = ["A", "B", "C", "D", "E"]
prompt_format = 'CQM-A'
prompter = Prompter("alpaca")

datasets_scienceqa = load_dataset('derek-thomas/ScienceQA')
save_path = "./data/scienceqa_RD_5/"
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)


datasets_wo_text_scienceqa = {'train': [], 'validation': [], 'test': []}
for split in splits:
    subject = {}
    topic = {}
    for n in tqdm(range(len(datasets_scienceqa[split]))):
        if datasets_scienceqa[split][n]['image'] == None and datasets_scienceqa[split][n]['topic'] not in bio_topics:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample['answer'] = answer
            data_sample['input'] = ""
            data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample['input']
            instruction = data_sample['instruction']
            output = data_sample['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample['text_full'] = full_prompt
            data_sample['text_input'] = input_prompt
            datasets_wo_text_scienceqa[split].append(data_sample)

    with open(save_path + 'scienceqa_not_biology_' + split + '_RD.json', 'w') as json_file:
        json.dump(datasets_wo_text_scienceqa[split], json_file)
    datasets_wo_text_scienceqa[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_wo_text_scienceqa[split]))
datasets_wo_text_scienceqa = datasets.DatasetDict(datasets_wo_text_scienceqa)
datasets_wo_text_scienceqa.save_to_disk(save_path + 'scienceqa_not_biology_all_RD.hf')
datasets_wo_text_scienceqa = load_from_disk(save_path + 'scienceqa_not_biology_all_RD.hf')
print(datasets_wo_text_scienceqa)
print(datasets_wo_text_scienceqa["validation"][0])

datasets_wo_text_scienceqa = {'train': [], 'validation': [], 'test': []}
for split in splits:
    subject = {}
    topic = {}
    for n in tqdm(range(len(datasets_scienceqa[split]))):
        if datasets_scienceqa[split][n]['image'] == None and datasets_scienceqa[split][n]['topic'] not in bio_phy_topics:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample['answer'] = answer
            data_sample['input'] = ""
            data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample['input']
            instruction = data_sample['instruction']
            output = data_sample['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample['text_full'] = full_prompt
            data_sample['text_input'] = input_prompt
            datasets_wo_text_scienceqa[split].append(data_sample)

    with open(save_path + 'scienceqa_not_biology_physics_' + split + '_RD.json', 'w') as json_file:
        json.dump(datasets_wo_text_scienceqa[split], json_file)
    datasets_wo_text_scienceqa[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_wo_text_scienceqa[split]))
datasets_wo_text_scienceqa = datasets.DatasetDict(datasets_wo_text_scienceqa)
datasets_wo_text_scienceqa.save_to_disk(save_path + 'scienceqa_not_biology_physics_all_RD.hf')
datasets_wo_text_scienceqa = load_from_disk(save_path + 'scienceqa_not_biology_physics_all_RD.hf')
print(datasets_wo_text_scienceqa)
print(datasets_wo_text_scienceqa["validation"][0])

datasets_wo_text_scienceqa = {'train': [], 'validation': [], 'test': []}
for split in splits:
    subject = {}
    topic = {}
    for n in tqdm(range(len(datasets_scienceqa[split]))):
        if datasets_scienceqa[split][n]['image'] == None and datasets_scienceqa[split][n]['topic'] not in bio_phy_che_topics:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample['answer'] = answer
            data_sample['input'] = ""
            data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample['input']
            instruction = data_sample['instruction']
            output = data_sample['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample['text_full'] = full_prompt
            data_sample['text_input'] = input_prompt
            datasets_wo_text_scienceqa[split].append(data_sample)

    with open(save_path + 'scienceqa_not_biology_physics_chemistry_' + split + '_RD.json', 'w') as json_file:
        json.dump(datasets_wo_text_scienceqa[split], json_file)
    datasets_wo_text_scienceqa[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_wo_text_scienceqa[split]))
datasets_wo_text_scienceqa = datasets.DatasetDict(datasets_wo_text_scienceqa)
datasets_wo_text_scienceqa.save_to_disk(save_path + 'scienceqa_not_biology_physics_chemistry_all_RD.hf')
datasets_wo_text_scienceqa = load_from_disk(save_path + 'scienceqa_not_biology_physics_chemistry_all_RD.hf')
print(datasets_wo_text_scienceqa)
print(datasets_wo_text_scienceqa["validation"][0])

datasets_wo_text_scienceqa = {'train': [], 'validation': [], 'test': []}
for split in splits:
    subject = {}
    topic = {}
    for n in tqdm(range(len(datasets_scienceqa[split]))):
        if datasets_scienceqa[split][n]['image'] == None and datasets_scienceqa[split][n]['topic'] not in bio_phy_che_eco_topics:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample['answer'] = answer
            data_sample['input'] = ""
            data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample['input']
            instruction = data_sample['instruction']
            output = data_sample['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample['text_full'] = full_prompt
            data_sample['text_input'] = input_prompt
            datasets_wo_text_scienceqa[split].append(data_sample)

    with open(save_path + 'scienceqa_not_biology_physics_chemistry_economics_' + split + '_RD.json', 'w') as json_file:
        json.dump(datasets_wo_text_scienceqa[split], json_file)
    datasets_wo_text_scienceqa[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_wo_text_scienceqa[split]))
datasets_wo_text_scienceqa = datasets.DatasetDict(datasets_wo_text_scienceqa)
datasets_wo_text_scienceqa.save_to_disk(save_path + 'scienceqa_not_biology_physics_chemistry_economics_all_RD.hf')
datasets_wo_text_scienceqa = load_from_disk(save_path + 'scienceqa_not_biology_physics_chemistry_economics_all_RD.hf')
print(datasets_wo_text_scienceqa)
print(datasets_wo_text_scienceqa["validation"][0])

datasets_wo_text_scienceqa = {'train': [], 'validation': [], 'test': []}
for split in splits:
    subject = {}
    topic = {}
    for n in tqdm(range(len(datasets_scienceqa[split]))):
        if datasets_scienceqa[split][n]['image'] == None and datasets_scienceqa[split][n]['topic'] not in bio_phy_che_eco_ear_topics:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample['answer'] = answer
            data_sample['input'] = ""
            data_sample['instruction'], data_sample['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample['input']
            instruction = data_sample['instruction']
            output = data_sample['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample['text_full'] = full_prompt
            data_sample['text_input'] = input_prompt
            datasets_wo_text_scienceqa[split].append(data_sample)

    with open(save_path + 'scienceqa_not_biology_physics_chemistry_economics_earth-science_' + split + '_RD.json', 'w') as json_file:
        json.dump(datasets_wo_text_scienceqa[split], json_file)
    datasets_wo_text_scienceqa[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_wo_text_scienceqa[split]))
datasets_wo_text_scienceqa = datasets.DatasetDict(datasets_wo_text_scienceqa)
datasets_wo_text_scienceqa.save_to_disk(save_path + 'scienceqa_not_biology_physics_chemistry_economics_earth-science_all_RD.hf')
datasets_wo_text_scienceqa = load_from_disk(save_path + 'scienceqa_not_biology_physics_chemistry_economics_earth-science_all_RD.hf')
print(datasets_wo_text_scienceqa)
print(datasets_wo_text_scienceqa["validation"][0])