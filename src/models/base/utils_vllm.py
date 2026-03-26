#    Copyright 2026 Federico Marcuzzi, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version, InvalidVersion

try:
    VLLM_VERSION_str = version("vllm")
    VLLM_VERSION = Version(VLLM_VERSION_str)
except (PackageNotFoundError, InvalidVersion):
    VLLM_VERSION = None

if VLLM_VERSION and VLLM_VERSION <= Version("0.7.2"):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


import torch
import numpy as np
from multiprocessing import Queue
import time

# VLLM 0.7.2 Logits Retriever
class VLLMLogitsRetriever:
    def __init__(self, tokenizer, options_map):
        self.base_to_toks = {base : [] for base in options_map.keys()}
        self.id_to_tok = {}
        self.tok_to_base = {}

        for base, toks in options_map.items():
            for tok in toks:
                id = tokenizer.encode(tok, add_special_tokens=False)
                if len(id) > 1:
                    print(f"[Info] - Option <{tok}> has more than one token id: {id}. Excluded")
                    continue
                else:
                    self.base_to_toks[base].append(tok)
                    self.id_to_tok[id[-1]] = tok
                    self.tok_to_base[tok] = base

        self.soft_max = {k : [] for k in self.base_to_toks.keys()}
        self.is_top = []
        self.pred = []
        self.prob = []
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        if token_ids == ():
            ids = list(self.id_to_tok.keys())
            ans_str = list(self.tok_to_base.keys())

            most_prop_tok = torch.argmax(logits)
            self.is_top.append(most_prop_tok in ids)

            soft_max = torch.nn.functional.softmax(logits, dim=0).to(dtype=float).cpu().numpy()
            sub_logits = soft_max[ids]
            self.pred.append(self.tok_to_base[ans_str[np.argmax(sub_logits)]])

            prob_sum = {k : 0 for k in self.base_to_toks.keys()}
            for ans, sm in zip(ans_str, sub_logits):
                prob_sum[self.tok_to_base[ans]] += sm
                
            for k, v in prob_sum.items():
                self.soft_max[k].append(v)

            self.prob.append(sum(prob_sum.values()))

        return logits

    def get_answers(self):
        return self.pred
    
    def get_soft_max(self):
        return self.soft_max
    
    def get_all(self):
        return {
            "answers" : self.pred,
            "soft_max" : self.soft_max,
            "prob_sum" : self.prob,
            "is_top" : self.is_top
        }


# VLLM 0.15.0 Logits Retriever
if VLLM_VERSION > Version("0.7.2"):
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, RequestLogitsProcessor
    from vllm import SamplingParams

    class VLLMLogitsRetrieverNew:
        queue = Queue()

        def __init__(self, tokenizer, options_map, prompts):
            self.base_to_toks = {base : [] for base in options_map.keys()}
            self.id_to_tok = {}
            self.tok_to_base = {}
            self.n_prompts = len(prompts)

            for base, toks in options_map.items():
                for tok in toks:
                    id = tokenizer.encode(tok, add_special_tokens=False)
                    if len(id) > 1:
                        print(f"[Info] - Option <{tok}> has more than one token id: {id}. Excluded")
                        continue
                    else:
                        self.base_to_toks[base].append(tok)
                        self.id_to_tok[id[-1]] = tok
                        self.tok_to_base[tok] = base

            self.soft_max = {k : [] for k in self.base_to_toks.keys()}
            self.is_top = []
            self.pred = []
            self.prob = []
            self.tokenizer = tokenizer

        def compute_data(self):
            start_time = time.time()
            last_print = start_time

            self.all_logits = {}
            while len(self.all_logits) < self.n_prompts:
                if VLLMLogitsRetrieverNew.queue.empty():
                    now = time.time()
                    if now - last_print >= 10:
                        waited = int(now - start_time)
                        print(f"Waiting for logits... {waited} seconds elapsed", len(self.all_logits), self.n_prompts)
                        last_print = now
                    continue
                prpmpt_id, logits = VLLMLogitsRetrieverNew.queue.get()
                self.all_logits[prpmpt_id] = logits

            for id in np.arange(self.n_prompts):
                self._data(self.all_logits[id])

        def _data(self, logits):
            ids = list(self.id_to_tok.keys())
            ans_str = list(self.tok_to_base.keys())

            most_prop_tok = torch.argmax(logits)
            self.is_top.append(most_prop_tok in ids)

            soft_max = torch.nn.functional.softmax(logits, dim=0).to(dtype=float).cpu().numpy()
            sub_logits = soft_max[ids]
            self.pred.append(self.tok_to_base[ans_str[np.argmax(sub_logits)]])

            prob_sum = {k : 0 for k in self.base_to_toks.keys()}
            for ans, sm in zip(ans_str, sub_logits):
                prob_sum[self.tok_to_base[ans]] += sm
                
            for k, v in prob_sum.items():
                self.soft_max[k].append(v)

            self.prob.append(sum(prob_sum.values()))
        
        def get_answers(self):
            return self.pred
        
        def get_soft_max(self):
            return self.soft_max
        
        def get_all(self):
            return {
                "answers" : self.pred,
                "soft_max" : self.soft_max,
                "prob_sum" : self.prob,
                "is_top" : self.is_top
            }
        
        def get_sample_params_list(self, sp_params, new_params):
            list_sp = []
            for prompt_id in range(self.n_prompts):
                cp_sp_params = sp_params.copy()
                cp_sp_params.update(new_params)
                cp_sp_params.update({"extra_args" : {"prompt_id": prompt_id}})
                list_sp.append(SamplingParams(**cp_sp_params))        
            return list_sp

    class PerReqLogitsRetriever:
        def __init__(self, queue, prompt_id: int) -> None:
            self.queue = queue
            self.prompt_id = prompt_id

        def __call__(self, output_ids: list[int], logits: torch.Tensor,) -> torch.Tensor:
            self.queue.put((self.prompt_id, logits.detach().cpu()))
            return logits

    class WrappedPerReqLogitsRetriever(AdapterLogitsProcessor):
        @classmethod
        def validate_params(cls, params: SamplingParams):
            prompt_id = params.extra_args and params.extra_args.get("prompt_id")
            if prompt_id is not None and not isinstance(prompt_id, int):
                raise ValueError(f"`prompt_id` has to be an integer, got {prompt_id}.")

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            super().__init__(vllm_config, device, is_pin_memory)
            self.is_cuda = device.type == "cuda"
            self.queue = VLLMLogitsRetrieverNew.queue

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(self, params: SamplingParams,) -> RequestLogitsProcessor | None:
            if (not self.is_cuda or (prompt_id := params.extra_args and params.extra_args.get("prompt_id")) is None):
                return None
            return PerReqLogitsRetriever(self.queue, prompt_id)