import time, sys, os

import torch
sys.path.append('syncode') # Assuming we are in the root directory
# print(os.getcwd())
from syncode import Syncode
import warnings
warnings.filterwarnings('ignore')

model_name = "microsoft/phi-2"

def test_syncode_spped_on_json():
    # Load the unconstrained original model
    llm = Syncode(model=model_name, mode='original', max_new_tokens=128)

    # Load the Syncode augmented model
    syn_llm = Syncode(model=model_name, grammar='json', parse_output_only=True, max_new_tokens=128, parser='lr')

    prompts = ['A JSON file describing a person:', 'A JSON file of a person John Smith:', 'A JSON file of a person John Smith with friends', 'JSON of a person Jane Doe with friends', 'A JSON person:']

    runs = 1
    llm_times = []
    syn_llm_times = []

    for _ in range(runs):
        for prompt in prompts:
            time_start = time.time()
            output = llm.infer(prompt)[0]
            # print(f"LLM output:\n{output}\n")
            llm_times.append(time.time() - time_start)

            time_start = time.time()
            output = syn_llm.infer(prompt)[0]
            # print(f"Syncode output:\n{output}\n")
            syn_llm_times.append(time.time() - time_start)

    print(f"LLM times: {llm_times}")
    print(f"Syncode times: {syn_llm_times}")
    print(f"Average LLM time: {sum(llm_times) / len(llm_times)}")
    print(f"Average Syncode time: {sum(syn_llm_times) / len(syn_llm_times)}")
    print(f"Speedup: {sum(llm_times) / sum(syn_llm_times)}")
    print(f"Overhead: {sum(syn_llm_times) / sum(llm_times)}")


def test_syncode_spped_on_c():
    # Load the unconstrained original model
    llm = Syncode(model=model_name, mode='original', max_new_tokens=128)

    # Load the Syncode augmented model
    syn_llm = Syncode(model=model_name, grammar='c', parse_output_only=True, max_new_tokens=128, parser='lr')

    prompts = ['A C program that prints "Hello, world!":\n```c\n', 'A C main function that iterates over an array of integers and prints each one:\n```c\n', 'A C program that prints the sum of two integers:\n```c\n', 'The following is a program that finds the sum of two integers in C:\n```c\n', 'A C program that fills an array with the numbers 0 to 9 and prints them:\n```c\n', 'A C implementation of a simple bubble sort:\n```c\n']

    runs = 1
    llm_times = []
    syn_llm_times = []

    for _ in range(runs):
        for prompt in prompts:
            # time_start = time.time()
            # output = llm.infer(prompt)[0]
            # print(f"LLM output:\n{output}\n")
            # llm_times.append(time.time() - time_start)

            # time_start = time.time()
            output = syn_llm.infer(prompt)[0]
            # print(f"Syncode output:\n{output}\n")
            # syn_llm_times.append(time.time() - time_start)

    # print(f"LLM times: {llm_times}")
    # print(f"Syncode times: {syn_llm_times}")
    # print(f"Average LLM time: {sum(llm_times) / len(llm_times)}")
    # print(f"Average Syncode time: {sum(syn_llm_times) / len(syn_llm_times)}")
    # print(f"Speedup: {sum(llm_times) / sum(syn_llm_times)}")
    # print(f"Overhead: {sum(syn_llm_times) / sum(llm_times)}")

def test_syncode_spped_on_sql():
    # Load the unconstrained original model
    llm = Syncode(model=model_name, mode='original', max_new_tokens=128)

    # Load the Syncode augmented model
    syn_llm = Syncode(model=model_name, grammar='go', parse_output_only=True, max_new_tokens=128, parser='lalr')

    prompts = ['A SQL query that selects all columns from a table:', 'A SQL query that selects all columns from a table named "people":', 'A SQL query that selects the first and last name columns from a table named "people":', 'A SQL query that selects the first and last name columns from a table named "people" where the age is greater than 18:', 'A SQL query that selects the first and last name columns from a table named "people" where the age is greater']

    runs = 1
    llm_times = []
    syn_llm_times = []

    for _ in range(runs):
        for prompt in prompts:
            time_start = time.time()
            output = llm.infer(prompt)[0]
            print(f"LLM output:\n{output}\n")
            llm_times.append(time.time() - time_start)

            time_start = time.time()
            output = syn_llm.infer(prompt)[0]
            print(f"Syncode output:\n{output}\n")
            syn_llm_times.append(time.time() - time_start)
    
    print(f"LLM times: {llm_times}")
    print(f"Syncode times: {syn_llm_times}")
    print(f"Average LLM time: {sum(llm_times) / len(llm_times)}")
    print(f"Average Syncode time: {sum(syn_llm_times) / len(syn_llm_times)}")
    print(f"Speedup: {sum(llm_times) / sum(syn_llm_times)}")
    print(f"Overhead: {sum(syn_llm_times) / sum(llm_times)}")

college_grammar = r"""
        ?start: function " " "of" " " dept code 
        function: "instructor" | "students" | "capacity" |  "deptcode"  | "school" | "college"
        dept:  /[A-Z]{3}/   
        code: /[0-9]{3}/ 
    """

college_prompt = """Paraphrase the following sentences
Human: who teaches CSE101?
Assistant:instructor of CSE101
Human: how many students can enroll in PSY456?
Assistant:capacity of PSY456
Human: what's the department of BIO433?
Assistant:"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from synchromesh import LarkCompletionEngine, HuggingFaceModel, predict_constrained

def test_syncode_tiny():
    syn_llm = Syncode(model='/data/share/models/hugging_face/Llama-7b', grammar=college_grammar, parse_output_only=True, max_new_tokens=128, parser='lr', mode='grammar_mask', log_level=2)

    time_start = time.time()
    for i in range(10):
        print(syn_llm.infer(college_prompt)[0])
    print(f"Time taken by syncode: {time.time() - time_start:.2f}s")


def test_synchromesh_tiny():
    """This is a simple example of using a grammar for semantic parsing.

    Suppose the assistant's role is to take a natural language question and rewrite it
    as a string accepted by a background grammar. CSD will ensure that the language model's
    output will always parse, even though the model itself is not aware of the grammar."""

    num_samples = 10
    comp_engine = LarkCompletionEngine(college_grammar, 'start', False)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = '/data/share/models/hugging_face/Llama-7b'
    # HF_MODEL = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    lm = HuggingFaceModel(model, tokenizer=tokenizer, prompt_template=college_prompt, temperature=0.25)

    for i in range(num_samples):
        print(HF_MODEL, "prediction:", predict_constrained(comp_engine, lm , 1, True, stop_tokens=["\n"]))


def test_synchromesh2():
    with open(f'syncode/parsers/grammars/json_grammar.lark', "r") as file:
        grammar = file.read()

    prompts = ['A JSON file describing a person:', 'A JSON file of a person John Smith:', 'A JSON file of a person John Smith with friends', 'JSON of a person Jane Doe with friends', 'A JSON person:']

    num_samples = 10
    comp_engine = LarkCompletionEngine(grammar, 'start', False)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    for prompt in prompts:
        print("Prompt:", prompt)
        lm = HuggingFaceModel(model, tokenizer=tokenizer, prompt_template=prompt, temperature=0.25)
        for i in range(num_samples):
            print(HF_MODEL, "prediction:", predict_constrained(comp_engine, lm , 1, True, stop_tokens=["\n"]))

# test_synchromesh_tiny()
# test_syncode_tiny()

# test_syncode_spped_on_json()
test_syncode_spped_on_c()
# test_syncode_spped_on_sql()
    