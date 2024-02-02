"""
This file provides base class Infer for generating code with mxeval  
"""
import pytest
import torch
import transformers
import os
from typing import Optional, Literal
from pathlib import PurePath
from monitors4codegen.multilspy.language_server import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from tests.test_utils import create_test_context, is_cuda_available
from transformers import AutoTokenizer, AutoModelForCausalLM
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper
from tqdm import tqdm
from mxeval.data import write_jsonl, read_problems, get_data, get_examples
import time
import numpy as np
import fire


from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

import typing

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]

# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function 
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list[str], size=4):
    mini_batches = []
    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


pytest_plugins = ("pytest_asyncio",)




class Infer:
    """Infer class for running inference on a model.

        List of currently tested models:
        Llama models: "Llama-7b", "CodeLlama-7b", "CodeLlama-7b-Python", "Llama-13b"
        CodeGen models: "Salesforce/codegen-350M-multi", "Salesforce/codegen2-1b"
        Bigcode models: "bigcode/starcoderbase-1b", "bigcode/santacoder" (1.1b WIP)
        WizardLM models: "WizardLM/WizardCoder-1B-V1.0"
    """
    def __init__(
        self, 
        model: str = 'Salesforce/codegen-350M-multi',
        quantize: bool = True,
        gpu: int = 1,
        num_samples: int = 1,
        grammar: str = "java",
        dataset: Literal["mbxp", "multi-humaneval", "mathqa-x", "input"] = "multi-humaneval", 
        mode: Literal["original2", "mgd"] = "original2",
        max_new_tokens = 100
    ): 
        self.model = model
        self.quantize = quantize
        self.gpu = gpu
        self.num_samples = num_samples
        self.grammar = grammar
        self.dataset = dataset
        self.num_samples = num_samples
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.early_stop = 10
        


        self.initialize_context()

        self.device = f"cuda:{self.gpu}"
        llama_models = ["Llama-7b", "Llama-13b", "CodeLlama-7b", "CodeLlama-7b-Python"]
        if self.model not in llama_models:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(self.device)
            out_dir = f"results/{self.model}/{self.grammar}/{self.dataset}/"
            
        elif self.model in llama_models:
            model_location = "/share/models/hugging_face/" + self.model
            tokenizer = LlamaTokenizer.from_pretrained(model_location)
            model = LlamaForCausalLM.from_pretrained(model_location, torch_dtype=torch.bfloat16).eval().to(self.device)
            out_dir = f"results/{self.model}/{self.grammar}/{self.dataset}/"
        
        self.out_path = out_dir + 'samples_' + str(self.num_samples) + '_mode_' + str(self.mode) + "_eval.jsonl"
        self.times_path = out_dir + 'samples_' + str(self.num_samples) + '_mode_' + str(self.mode) + "_times.txt"
        os.makedirs(out_dir, exist_ok=True)
        
        if self.dataset != 'input':
            self.run_code_eval(model, tokenizer)
        
        else:
            self.user_input(model, tokenizer)

    def run_code_eval(self, model, tokenizer):
        samples = []
        problems = get_data(self.dataset, self.grammar)
        pbar = tqdm(total=len(problems) * self.num_samples)
        generation_times = []

        i = 0
        for task_id in problems:
            if i == self.early_stop:
                break
            prompt = problems[task_id]["prompt"]

            batch_completions, generation_time = self.batch_completion(model, tokenizer, prompt)

            generation_times.append(generation_time)

            for _, completion in enumerate(batch_completions):
                result = dict(
                    task_id=task_id,
                    language=problems[task_id]["language"],
                    completion=completion
                )
                samples += [result]
            
            pbar.update(self.num_samples)
            i += 1
        
        self.mean_generation = np.mean(generation_times)
        write_jsonl(self.out_path, samples)
    

    def initialize_context(self):
        if self.grammar == 'python':
            self.params = {
                "code_language": Language.PYTHON,
                "repo_url": "https://github.com/tarsur909/testfolder/",
                "repo_commit": "7c58c74072ce5457bdbf02aaf0f0a7cc75b332a1",
            }
            self.filepath = "day_10_synapses.py"
        
        elif self.grammar == 'java':
            self.params = {
                "code_language": Language.JAVA,
                "repo_url": "https://github.com/LakshyAAAgrawal/clickhouse-highlevel-sinker/",
                "repo_commit": "5775fd7a67e7b60998e1614cf44a8a1fc3190ab0"
            }
            self.filepath = "src/main/java/com/xlvchao/clickhouse/datasource/ClickHouseDataSource.java"


    def batch_completion(self, model, tokenizer, prompt):
        with create_test_context(self.params) as context:
                lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)

                with lsp.start_server():
                    with lsp.open_file(self.filepath):
                        new_lc = lsp.language_server.insert_text_at_position(self.filepath, 7, 0, prompt)

                        filebuffer = MonitorFileBuffer(lsp.language_server, self.filepath, (new_lc['line'], new_lc['character']), (new_lc['line'], new_lc['character']), self.params['code_language'])

                        logists_procesor_lst = None
                        if self.mode == 'mgd':
                        
                            monitor = DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                            mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)
                            logists_procesor_lst = LogitsProcessorList([mgd_logits_processor])
                        
 
                        input_batch = [prompt for _ in range(self.num_samples)]
                        inputs = tokenizer(input_batch, return_tensors="pt").to(self.device)
                        input_ids_cutoff = inputs.input_ids.size(dim=1)

                        start = time.time()
                        generated_ids = model.generate(
                            **inputs,
                            use_cache=True,
                            max_new_tokens= self.max_new_tokens,
                            temperature=0.2,
                            top_p=0.95,
                            do_sample=False,
                            eos_token_id= tokenizer.eos_token_id,
                            pad_token_id= tokenizer.eos_token_id, 
                            logits_processor= logists_procesor_lst
                        )

                        end = time.time() - start

                        completions = []
                        for i in range(self.num_samples):
                            completion = tokenizer.decode(generated_ids[i][input_ids_cutoff-1:],
                                        skip_special_tokens=True)[1:]
                            
                            if self.grammar == 'python':
                                completion = filter_code(fix_indents(completion))
                            
                            else:
                                completion = filter_code(completion)
                            
                            completions.append(completion)
                        
                        return completions, end


