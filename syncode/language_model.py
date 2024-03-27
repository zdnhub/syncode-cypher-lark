import time
from syncode.grammar_decoder import GrammarDecoder
from transformers import LogitsProcessorList
from syncode.parsers.grammars import Grammar
from syncode.utils.generation import filter_code, fix_indents
import syncode.common as common
import torch
from typing import Callable, Iterable, Tuple
from transformers.generation.utils import GenerationMode
from transformers.generation.configuration_utils import GenerationConfig

class LanguageModel:
    def vocabulary(self) -> Iterable[str]:
        raise NotImplementedError()

    def predict_tokens(self, prefix: str, n: int) -> Iterable[int]:
        raise NotImplementedError()

    def predict_token(self, prefix: str, valid_tokens: Iterable[int], top_k: int = 1) -> Tuple[Iterable[int], Iterable[float]]:
        'Given prefix (prompt + already generated code), predicts next token'
        raise NotImplementedError()

    def tokenize(self, s: str) -> Iterable[int]:
        raise NotImplementedError()

    def get_token(self, i: int) -> str:
        return self.vocabulary()[i]

    def predict_unconstrained(self, prefix: str, max_tokens: int, stop=None):
        raise NotImplementedError()


class HuggingFaceModel(LanguageModel):
    def __init__(
            self, 
            model: Callable, 
            grammar: Grammar,
            logger: common.Logger, 
            tokenizer=None, 
            prompt_template: str = '', 
            api_key: str = None, 
            best_of: int = 1, 
            before_prediction_hook=lambda: None, 
            device='cuda', 
            grammar_decoder=None, 
            mode: str ='original', 
            **kwargs) -> None:
        super().__init__()

        self.prompt_template = prompt_template
        self.model = model
        self.logger = logger
        self.tokenizer = tokenizer
        self.device = device
        self.best_of = best_of
        self._before_prediction_hook = before_prediction_hook
        self.grammar_decoder = grammar_decoder
        self.logit_processors: Iterable = LogitsProcessorList([self.grammar_decoder])
        self.mode = mode
        self.grammar = grammar
        self.vocab = common.get_vocab_from_tokenizer(self.tokenizer)
        self.gen_args = kwargs

    def get_grammar_decoder(self):
        if self.logit_processors is not None and len(self.logit_processors) > 0:
            return self.logit_processors[0]
        return None

    @torch.inference_mode()
    def generate_batch_completion_grammar(self, prompt, batch_size) -> Iterable[str]:
        '''
        Generates batch_size completions for the given prompt. 
        '''
        # Reset the grammar decoder
        if self.grammar_decoder is not None:
            self.grammar_decoder.reset(prompt)

        input_batch = [prompt for _ in range(batch_size)]
        inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.model.device)
        input_ids_cutoff = inputs.input_ids.size(dim=1)

        start_time = time.time()
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.update(**self.gen_args)

        gen_mode = self._get_generation_mode(gen_config)

        if (gen_mode == GenerationMode.SAMPLE or gen_mode == GenerationMode.GREEDY_SEARCH) and batch_size == 1: # Use our own implementation for greedy search and sampling
            generated_ids = self._generate(inputs, gen_config, gen_mode, grammar_decoder=self.grammar_decoder)
        else:
            # Use generate from transformers library for other modes
            generated_ids = self.model.generate(**inputs, logits_processor=self.logit_processors, **self.gen_args)
        batch_completions = []

        for i in range(batch_size):
            raw_completion = self.tokenizer.decode(generated_ids[i][input_ids_cutoff:len(generated_ids[i])], skip_special_tokens=True)                                       
            self.logger.log_code("Raw completion", raw_completion)

            # Post-processing to filter out using stop word (e.g. "\n\n")
            if self.grammar != None and self.grammar.name == "python":
                completion = self.postproces_completion_python(i, batch_size, input_ids_cutoff, generated_ids, self.grammar_decoder, raw_completion)
                self.logger.log_code("Filtered sample", completion)
            elif self.grammar != None and self.grammar.name == "go": 
                completion = self.postproces_completion_go(i, batch_size, raw_completion, generated_ids, self.grammar_decoder, input_ids_cutoff)
                self.logger.log_code("Filtered sample", completion)
            else: # TODO: handle the case for other grammars
                completion = raw_completion

            batch_completions.append(completion)

        if self.grammar_decoder is not None:
            self.logger.log_time(f"Time taken for generation: {time.time() - start_time:.2f}s")
        
        self.logger.log(f"Completion: {batch_completions}")
        return batch_completions

    @torch.inference_mode()
    def _generate(
        self, 
        inputs:dict, 
        gen_config:GenerationConfig, 
        gen_mode:GenerationMode, 
        grammar_decoder:GrammarDecoder=None
        ):
        """
        We support greedy search and sampling for batch size 1 otherwise we use the generate function from transformers library.
        """
        token_ids, attention_mask, past_key_values = inputs['input_ids'], inputs['attention_mask'], None
        logit_warper = self.model._get_logits_warper(gen_config)
        max_tokens = self.gen_args['max_new_tokens']+token_ids.size(1)
        time_inference = 0
        time_syncode = 0
        valid_count = 0
        total = 0

        while True:
            time1 = time.time()
            try:
                if past_key_values: # Get the last token if kv is cached for all previous tokens
                    input_ids = token_ids[..., -1].unsqueeze(-1) 
                else:
                    input_ids = token_ids

                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    past_key_values=past_key_values
                    )                
            except IndexError as e:  
                raise ValueError(f"The input length exceeds the context length of the model. {e}")

            next_token_scores, past_key_values = outputs.logits[:, -1, :], outputs.past_key_values
            
            time_inference += (time.time() - time1)
            time2 = time.time()

            if grammar_decoder is not None:
                next_token = self._get_next_token(gen_mode, token_ids, logit_warper, next_token_scores)
                is_valid = grammar_decoder.is_valid(token_ids, next_token)
                # print(is_valid)
                total += 1
                valid_count += is_valid

                if not is_valid:
                    # calling grammar decoder is expensive. Hence, in the opportunist mode, we call it only when the standard generation is syntactically incorrect
                    next_token_scores = grammar_decoder(token_ids, next_token_scores)
                    next_token = self._get_next_token(gen_mode, token_ids, logit_warper, next_token_scores)
            else:
                next_token = self._get_next_token(gen_mode, token_ids, logit_warper, next_token_scores)
            
            time_syncode += (time.time() - time2)

            token_ids = torch.cat([token_ids, next_token[:, None]], dim=-1)

            if next_token == self.tokenizer.eos_token_id or token_ids.size(1) >= max_tokens:
                break
            
            # Update attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(self.device)], dim=-1)
        
        # print(f"Time taken for inference: {time_inference:.2f}s")
        # print(f"Time taken for syncode: {time_syncode:.2f}s")
        # print(f"Valid count: {valid_count}/{total}")

        return token_ids

    def _get_next_token(self, gen_mode, token_ids, logit_warper, next_token_scores):
        if gen_mode == GenerationMode.GREEDY_SEARCH:
            next_token = torch.argmax(next_token_scores, dim=-1)
        elif gen_mode == GenerationMode.SAMPLE:
            next_token_scores = logit_warper(token_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token

    def _get_generation_mode(
        self, generation_config: GenerationConfig
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH
        return generation_mode

    @torch.inference_mode()
    def generate_chat_completion_grammar(self, prompt) -> str:
        '''
        Generates chat completion for the given prompt. 
        '''
        assert 'apply_chat_template' in dir(self.tokenizer), "Tokenizer does not support chat completion"

        message = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            message, 
            add_generation_prompt=True, 
            return_tensors="pt"
            ).to(self.model.device)
        
        input_ids_cutoff = inputs.size(dim=1)

        # input_ids_cutoff = inputs.input_ids.size(dim=1)
        start_time = time.time()

        generated_ids = self.model.generate(
            inputs,
            logits_processor=self.logit_processors,
            **self.gen_args
        )
        completion = self.tokenizer.decode(
            generated_ids[0][input_ids_cutoff:len(generated_ids[0])], 
            skip_special_tokens=True)

        self.logger.log_code("Raw completion", completion)
        return completion


    def postproces_completion_python(self, i, batch_size, input_ids_cutoff, generated_ids, grammar_decoder, raw_completion):
        stop_word = "\n\n"
        if stop_word in raw_completion or self.tokenizer.eos_token_id == generated_ids[i][-1] or grammar_decoder is None:
            completion = filter_code(fix_indents(raw_completion))
        else:
            # Use when the stop word does not exist in the completion and grammar_decoder is used
            function_incomplete = [False for _ in range(batch_size)]
            completion = self.compute_backup_completion(grammar_decoder, function_incomplete, i, input_ids_cutoff, generated_ids)
        return completion

    def postproces_completion_go(self, i, batch_size, raw_completion, generated_ids, grammar_decoder, input_ids_cutoff):
        extra_stop_word = "\n\n"
        
        if self.mode == "original": 
            # only filter with stop-word for original mode
            completion = filter_code(raw_completion, extra_stop_word=extra_stop_word)
        else:
            # When the grammar_decoder is used
            function_incomplete = [False for _ in range(batch_size)]
            self.logger.log(f"Function incomplete!")
            completion = self.compute_backup_completion(grammar_decoder, function_incomplete, i, input_ids_cutoff, generated_ids)

            if function_incomplete[i]:
                # if the function is incomplete, then we need to add a closing brace
                completion += "}"

        return completion

    def compute_backup_completion(self, grammar_decoder, function_incomplete, i, input_ids_cutoff, generated_ids):
        self.logger.log(f"Last valid state: {grammar_decoder.last_valid_state[i]}")
        self.logger.log(f"Function end: {grammar_decoder.function_end[i]}")

        if grammar_decoder.function_end[i] is not None:
            # if the function end is not None, then the last valid state is the function end
            last_token_id = grammar_decoder.function_end[i]
        else:
            # otherwise, the last valid state is the last valid state
            function_incomplete[i] = True
            last_token_id = grammar_decoder.last_valid_state[i]
        # return last_token_id
        # Use when the stop word does not exist in the completion
        backup_completion = self.tokenizer.decode(generated_ids[i][input_ids_cutoff-1:last_token_id],
        skip_special_tokens=True)[1:]  
        return backup_completion
    
    def tokenize(self, s: str) -> 'Iterable[int]':
        return self.tokenizer.encode(s, add_special_tokens=False)

    def vocabulary(self) -> Iterable[str]:
        return self.vocab

    def predict_token(self, prefix: str, valid_tokens: Iterable[int], top_k: int = 1) -> Tuple[Iterable[int], Iterable[float]]:
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, -1]
            valid_tokens_mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
            valid_tokens_set = set(valid_tokens)
            if None in valid_tokens_set:
                valid_tokens_set.remove(None)
            valid_tokens = list(valid_tokens_set)
            # make sure there's no None in valid_tokens
            valid_tokens_mask[valid_tokens] = True
            logits = logits.squeeze(0)
            filtered_logits = logits.softmax(dim=-1)
            filtered_logits[~valid_tokens_mask] = 0
            filtered_logits = filtered_logits / filtered_logits.sum()

        top_values, top_indices = torch.topk(filtered_logits, top_k)
        top_indices = top_indices.tolist()
        top_values = top_values.log().tolist()
        
        return top_indices, top_values

    def predict_unconstrained(self, prefix: str, max_tokens: int, stop=None):
        prompt = f"{self.prompt_template}{prefix}"
        print('Prompt:', prompt)
        print(repr(prompt))
        
        self._before_prediction_hook()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

        # Cut off inputs which are too long
        input_length = self.model.config.max_position_embeddings - max_tokens - 1
        if len(input_ids[0]) > input_length:
            input_ids = input_ids[:, -input_length:]
        input_ids = input_ids.to(self.device)
        with torch.cuda.amp.autocast():
            if self.temperature == 0.0:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            # remove the prompt
            output = output[:, len(input_ids[0]):]
        # print('Here is the output:', output[0])
        # detokenized = self.tokenizer.decode(output[0])
        # if stop is not None:
        #     for stop_token in stop:
        #         if stop_token in detokenized:
        #             # split on the first stop token
        #             detokenized = detokenized.split(stop_token)[0]
        return output[0]
    