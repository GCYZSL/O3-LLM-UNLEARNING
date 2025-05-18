import json
from datasets import load_from_disk, load_dataset
import datasets
import pandas as pd
from typing import Union

datasets_scienceqa = load_dataset('derek-thomas/ScienceQA')
datasets_commonqa = load_dataset('tau/commonsense_qa')
datasets_openbookqa = load_dataset('openbookqa', 'main')


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

prompter = Prompter("alpaca")

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

def get_choice_text_coqa(problem, options):
    choices = problem['label']
    texts = problem['text']
    choice_list = []
    for i in range(len(choices)):
        choice_list.append("({}) {}".format(choices[i], texts[i]))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
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

def create_one_example_coqa(format, question, choice, answer, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Question: {question}\nOptions: {choice}\n"
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
        # output = f"Answer: The answer is ({answer})."
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

datasets_all = {'train': [], 'validation': [], 'test': []}
datasets_scienceqa_our = {'train': [], 'validation': [], 'test': []}
datasets_commonqa_our = {'train': [], 'validation': [], 'test': []}
datasets_openbookqa_our = {'train': [], 'validation': [], 'test': []}
splits = ['test', 'validation', 'train']

subject = {}
topic = {}

options = ["A", "B", "C", "D", "E"]
prompt_format = 'CQM-A'
import os
if not os.path.exists("./data/scienceqa"):
    os.makedirs("./data/scienceqa")
if not os.path.exists("./data/commonqa"):
    os.makedirs("./data/commonqa")
if not os.path.exists("./data/openbookqa"):
    os.makedirs("./data/openbookqa")
if not os.path.exists("./data/qa_all"):
    os.makedirs("./data/qa_all")

### SCIENCEQA
for split in splits:
    for n in range(len(datasets_scienceqa[split])):
        if datasets_scienceqa[split][n]['image'] == None:
            data_sample = datasets_scienceqa[split][n]
            question = datasets_scienceqa[split][n]['question']
            context = get_context_text(datasets_scienceqa[split][n], False)
            choice = get_choice_text(datasets_scienceqa[split][n], options)
            answer = get_answer(datasets_scienceqa[split][n], options)
            lecture = get_lecture_text(datasets_scienceqa[split][n])
            solution = get_solution_text(datasets_scienceqa[split][n])

            data_sample_s = {}
            data_sample_s['answer'] = answer
            data_sample_s['input'] = ""
            data_sample_s['instruction'], data_sample_s['output'], _ = create_one_example(prompt_format,
                                                                                      question,
                                                                                      context,
                                                                                      choice,
                                                                                      answer,
                                                                                      lecture,
                                                                                      solution,
                                                                                      test_example=False)
            input = data_sample_s['input']
            instruction = data_sample_s['instruction']
            output = data_sample_s['output']
            full_prompt = prompter.generate_prompt(
                instruction,
                input,
                output,
            )

            input_prompt = prompter.generate_prompt(
                instruction,
                input,
            )
            data_sample_s['text_full'] = full_prompt
            data_sample_s['text_input'] = input_prompt

            datasets_all[split].append(data_sample_s)
            datasets_scienceqa_our[split].append(data_sample_s)

    with open('./data/scienceqa/scienceqa_' + split + '.json', 'w') as json_file:
        json.dump(datasets_scienceqa_our[split], json_file)
    datasets_scienceqa_our[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_scienceqa_our[split]))
datasets_scienceqa_our = datasets.DatasetDict(datasets_scienceqa_our)
datasets_scienceqa_our.save_to_disk('./data/scienceqa/scienceqa_all.hf')
datasets_scienceqa_our = load_from_disk('./data/scienceqa/scienceqa_all.hf')
print(datasets_scienceqa_our)
print(datasets_scienceqa_our['train'][0])


###COMMONQA
type = 'commonq_'
splits = ['validation', 'train']
for split in splits:
    print(split)
    for n in range(len(datasets_commonqa[split])):
        data_sample = datasets_commonqa[split][n]
        question = datasets_commonqa[split][n]['question']
        question_concept = datasets_commonqa[split][n]['question_concept']
        choice = get_choice_text_coqa(datasets_commonqa[split][n]["choices"], options)
        answer = datasets_commonqa[split][n]['answerKey']

        data_sample_s = {}
        data_sample_s['answer'] = answer
        data_sample_s['input'] = ""
        data_sample_s['instruction'], data_sample_s['output'], _ = create_one_example_coqa(prompt_format,
                                                                                  question,
                                                                                  choice,
                                                                                  answer,
                                                                                  test_example=False)
        input = data_sample_s['input']
        instruction = data_sample_s['instruction']
        output = data_sample_s['output']
        full_prompt = prompter.generate_prompt(
            instruction,
            input,
            output,
        )

        input_prompt = prompter.generate_prompt(
            instruction,
            input,
        )
        data_sample_s['text_full'] = full_prompt
        data_sample_s['text_input'] = input_prompt

        datasets_all[split].append(data_sample_s)
        datasets_commonqa_our[split].append(data_sample_s)

    with open('./data/commonqa/commonqa_' + split + '.json', 'w') as json_file:
        json.dump(datasets_commonqa_our[split], json_file)
    datasets_commonqa_our[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_commonqa_our[split]))

split = "test"
print(split)
for n in range(len(datasets_commonqa["validation"])):
    data_sample = datasets_commonqa["validation"][n]
    question = datasets_commonqa["validation"][n]['question']
    question_concept = datasets_commonqa["validation"][n]['question_concept']
    choice = get_choice_text_coqa(datasets_commonqa["validation"][n]["choices"], options)
    answer = datasets_commonqa["validation"][n]['answerKey']

    data_sample_s = {}
    data_sample_s['answer'] = answer
    data_sample_s['input'] = ""
    data_sample_s['instruction'], data_sample_s['output'], _ = create_one_example_coqa(prompt_format,
                                                                              question,
                                                                              choice,
                                                                              answer,
                                                                              test_example=False)
    input = data_sample_s['input']
    instruction = data_sample_s['instruction']
    output = data_sample_s['output']
    full_prompt = prompter.generate_prompt(
        instruction,
        input,
        output,
    )

    input_prompt = prompter.generate_prompt(
        instruction,
        input,
    )
    data_sample_s['text_full'] = full_prompt
    data_sample_s['text_input'] = input_prompt

    datasets_all[split].append(data_sample_s)
    datasets_commonqa_our[split].append(data_sample_s)

with open('./data/commonqa/commonqa_' + split + '.json', 'w') as json_file:
    json.dump(datasets_commonqa_our[split], json_file)
datasets_commonqa_our[split] = datasets.Dataset.from_pandas(
    pd.DataFrame(data=datasets_commonqa_our[split]))

datasets_commonqa_our = datasets.DatasetDict(datasets_commonqa_our)
datasets_commonqa_our.save_to_disk('./data/commonqa/commonqa_all.hf')
datasets_commonqa_our = load_from_disk('./data/commonqa/commonqa_all.hf')
print(datasets_commonqa_our)
print(datasets_commonqa_our['train'][0])

### OPENBOOKQA
splits = ['test', 'validation', 'train']
for split in splits:
    print(split)
    for n in range(len(datasets_openbookqa[split])):
        data_sample = datasets_openbookqa[split][n]
        question = datasets_openbookqa[split][n]['question_stem']
        choice = get_choice_text_coqa(datasets_openbookqa[split][n]["choices"], options)
        answer = datasets_openbookqa[split][n]['answerKey']

        data_sample_s = {}
        data_sample_s['answer'] = answer
        data_sample_s['input'] = ""
        data_sample_s['instruction'], data_sample_s['output'], _ = create_one_example_coqa(prompt_format,
                                       question,
                                       choice,
                                       answer,
                                       test_example=False)

        input = data_sample_s['input']
        instruction = data_sample_s['instruction']
        output = data_sample_s['output']
        full_prompt = prompter.generate_prompt(
            instruction,
            input,
            output,
        )

        input_prompt = prompter.generate_prompt(
            instruction,
            input,
        )
        data_sample_s['text_full'] = full_prompt
        data_sample_s['text_input'] = input_prompt

        datasets_all[split].append(data_sample_s)
        datasets_openbookqa_our[split].append(data_sample_s)

    with open('./data/openbookqa/openbookqa_' + split + '.json', 'w') as json_file:
        json.dump(datasets_openbookqa_our[split], json_file)
    datasets_openbookqa_our[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_openbookqa_our[split]))

    with open('./data/qa_all/qaall_' + split + '.json', 'w') as json_file:
        json.dump(datasets_all[split], json_file)
    datasets_all[split] = datasets.Dataset.from_pandas(
        pd.DataFrame(data=datasets_all[split]))

datasets_openbookqa_our = datasets.DatasetDict(datasets_openbookqa_our)
datasets_openbookqa_our.save_to_disk('./data/openbookqa/openbookqa_all.hf')
datasets_openbookqa_our = load_from_disk('./data/openbookqa/openbookqa_all.hf')
print(datasets_openbookqa_our)
print(datasets_openbookqa_our['train'][0])

datasets_all = datasets.DatasetDict(datasets_all)
datasets_all.save_to_disk('./data/qa_all/qaall_all.hf')
datasets_all = load_from_disk('./data/qa_all/qaall_all.hf')
print(datasets_all)
print(datasets_all['train'][0])
