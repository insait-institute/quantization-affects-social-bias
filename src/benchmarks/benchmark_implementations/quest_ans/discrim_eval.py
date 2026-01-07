#    Copyright 2025 Federico Marcuzzi, INSAIT
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


from typing import cast

import pandas as pd

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics

from datasets import load_from_disk
from src.benchmarks.benchmark_implementations.utils import *


MAP = {"Yes": 0, "No": 1}

ANS_MAP_YN = {
                "Yes" : ["Yes", " Yes", "Yes ", " Yes ", "yes", " yes", "yes ", " yes ", "YES", " YES", "YES ", " YES "],
                "No" : ["No", " No", "No ", " No ", "no", " no", "no ", " no ", "NO", " NO", "NO ", " NO "],
            }

class DiscrimEvalConfig(BenchmarkConfig):
    pass

class DiscrimEvalDataConfig(DataConfig):
    pass

class DiscrimEvalDataProvider(BaseData):

    def _get_promt(self, prompt):
        return prompt + "\n\nMy answer would be "

    def __init__(self, data_context: BaseDataContext):
        print(f"[INFO] DiscrimEval: init")

        data_config = data_context.get_data_config()
        self.config = data_config
        self.debug = data_config.debug
        self.subset_size = data_config.subset_size

        """
            get DescrimEval dataset from https://huggingface.co/datasets/Anthropic/discrim-eval - explicit version
        """
        data = load_from_disk("benchmark_datasets/discrim_eval/train").to_pandas()

        debug_size = self.subset_size if self.debug else len(data)
        real_subset_size = min(len(data), debug_size)
        data.drop(data.index[real_subset_size:], inplace=True)

        data['final_prompt'] = data["filled_template"].apply(self._get_promt)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data

class DiscrimEval(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = self.context.get_benchmark_config()
        self.data_provider = cast(DiscrimEvalDataProvider, self.context.get_dataset())

    def run(self, model: BaseModel):
        print(f"[INFO] DiscrimEval: run")

        model.reset_statistic()
        data = self.data_provider.get_data()

        PromptStatistics.reset()
        output = model.most_prob_options(data["final_prompt"].to_list(), ANS_MAP_YN, get_soft_max=True)
        PromptStatistics.dump("DiscrimEval")

        results = {
            "aggregated_results" : {"token_statistics" : model.get_statistic()},
            "raw_results" : {
                "yes_prob" : output["Yes"],
                "no_prob" : output["No"],
                "decision_question_id" : data["decision_question_id"].tolist(),
                "race" : data["race"].tolist(),
                "gender" : data["gender"].tolist(),
                "age" : data["age"].tolist(),
                "prompts" : data["final_prompt"].tolist(),
                "filled_template" : data["filled_template"].tolist(),
                "fill_type" : data["fill_type"].tolist(),
                "prompts" : data["final_prompt"].tolist(),
            },
            "benchmark_params" : {"answer_map" : MAP},
        }

        return to_serializable(results)