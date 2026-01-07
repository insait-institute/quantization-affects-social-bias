import os
from pathlib import Path

from helper_tools.results_processor import (
    reformat_generic,
    reformat_counterfactual_sentences,
    reformat_mmlu_full,
    reformat_coreference_resolution,
    reformat_bbq,
    reformat_sentence_completion,
    reformat_discrim_eval,
    reformat_discrim_eval_gen,
    reformat_dt_fairness,
)

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.base_data import BaseData
from src.metrics.base_metric import BaseMetric
from src.registry import ComponentRegistry
from src.registry import registry
from src.registry import BENCHMARK_PROCESSORS

# Import all the benchmark implementations
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.configs.base_metric_config import MetricConfig


####
# General global constants
####

# Directory from where to include other configs
CONFIG_DIR = os.path.abspath("configs/")
RESULTS_DIR = os.path.abspath("results/")
CODE_ROOT_PATH = Path(os.path.abspath(__file__)) / "src"

metric_registry = ComponentRegistry(BaseMetric, MetricConfig)
dataset_registry = ComponentRegistry(BaseData, DataConfig)
benchmark_registry = ComponentRegistry(BaseBenchmark, BenchmarkConfig)

registry.register("metric", metric_registry)
registry.register("data", dataset_registry)
registry.register("benchmark", benchmark_registry)


###
# ACCURACY
###

from src.benchmarks.benchmark_implementations.quest_ans import mmlu_full

benchmark_registry.register_logic_config_classes(
    "mmlu_full",
    mmlu_full.MMLU,
    mmlu_full.MMLUConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "mmlu_full_data",
    mmlu_full.MMLUDataProvider,
    mmlu_full.MMLUDataConfig,
)


###
# COUNTERFACTUAL SENTENCES
###

from src.benchmarks.benchmark_implementations.counter_sents import reddit_bias

benchmark_registry.register_logic_config_classes(
    "reddit_bias",
    reddit_bias.RedditBias,
    reddit_bias.RedditBiasConfig,
    category="counter_sents"
)

dataset_registry.register_logic_config_classes(
    "reddit_bias_data",
    reddit_bias.RedditBiasDataProvider,
    reddit_bias.RedditBiasDataConfig,
)

from src.benchmarks.benchmark_implementations.counter_sents import stereo_set

benchmark_registry.register_logic_config_classes(
    "stereo_set",
    stereo_set.StereoSet,
    stereo_set.StereoSetConfig,
    category="counter_sents"
)

dataset_registry.register_logic_config_classes(
    "stereo_set_data",
    stereo_set.StereoSetDataProvider,
    stereo_set.StereoSetDataConfig,
)


###
# SENTENCE COMPLETION
###

from src.benchmarks.benchmark_implementations.sents_compl import bold

benchmark_registry.register_logic_config_classes(
    "bold",
    bold.Bold,
    bold.BoldConfig,
    category="sents_compl",
)

dataset_registry.register_logic_config_classes(
    "bold_data",
    bold.BoldDataProvider,
    bold.BoldDataConfig,
)

from src.benchmarks.benchmark_implementations.sents_compl import dt_toxic

benchmark_registry.register_logic_config_classes(
    "dt_toxic",
    dt_toxic.DecodigTrustToxicity,
    dt_toxic.DecodigTrustToxicityConfig,
    category="sents_compl",
)

dataset_registry.register_logic_config_classes(
    "dt_toxic_data",
    dt_toxic.DecodigTrustToxicityDataProvider,
    dt_toxic.DecodigTrustToxicityDataConfig,
)


####
# QUESTION ANSWERING
####

from src.benchmarks.benchmark_implementations.quest_ans import wino_bias

benchmark_registry.register_logic_config_classes(
    "wino_bias",
    wino_bias.WinoBias,
    wino_bias.WinoBiasConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "wino_bias_data",
    wino_bias.WinoBiasDataProvider,
    wino_bias.WinoBiasDataConfig,
)

from src.benchmarks.benchmark_implementations.quest_ans import bbq

benchmark_registry.register_logic_config_classes(
    "bbq",
    bbq.BBQ,
    bbq.BBQConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "bbq_data",
    bbq.BBQDataProvider,
    bbq.BBQDataConfig,
)

from src.benchmarks.benchmark_implementations.quest_ans import dt_fairness

benchmark_registry.register_logic_config_classes(
    "dt_fairness",
    dt_fairness.DecodingTrustFairness,
    dt_fairness.DecodingTrustFairnessConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "dt_fairness_data",
    dt_fairness.DecodingTrustFairnessDataProvider,
    dt_fairness.DecodingTrustFairnessDataConfig,
)

from src.benchmarks.benchmark_implementations.quest_ans import discrim_eval_gen

benchmark_registry.register_logic_config_classes(
    "discrim_eval_gen",
    discrim_eval_gen.DiscrimEvalGen,
    discrim_eval_gen.DiscrimEvalGenConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "discrim_eval_gen_data",
    discrim_eval_gen.DiscrimEvalGenDataProvider,
    discrim_eval_gen.DiscrimEvalGenDataConfig,
)

from src.benchmarks.benchmark_implementations.quest_ans import discrim_eval

benchmark_registry.register_logic_config_classes(
    "discrim_eval",
    discrim_eval.DiscrimEval,
    discrim_eval.DiscrimEvalConfig,
    category="quest_ans",
)

dataset_registry.register_logic_config_classes(
    "discrim_eval_data",
    discrim_eval.DiscrimEvalDataProvider,
    discrim_eval.DiscrimEvalDataConfig,
)


####
# Benchmark Postprocessing
####
BENCHMARK_PROCESSORS |= {
    "reformat_generic": reformat_generic,
    "reformat_counterfactual_sentences": reformat_counterfactual_sentences,
    "reformat_mmlu_full" : reformat_mmlu_full,
    "reformat_coreference_resolution": reformat_coreference_resolution,
    "reformat_bbq": reformat_bbq,
    "reformat_sentence_completion": reformat_sentence_completion,
    "reformat_discrim_eval": reformat_discrim_eval,
    "reformat_discrim_eval_gen": reformat_discrim_eval_gen,
    "reformat_dt_fairness": reformat_dt_fairness,
}