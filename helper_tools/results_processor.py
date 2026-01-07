#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#    Modifications Copyright 2025 Federico Marcuzzi, INSAIT
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

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import copy
import numpy as np
import pandas as pd
from itertools import combinations

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# These imports need to be after the path context import, otherwise the imported modules are not found.
from src.results.base_results_repository import BaseResultsRepository, RunInfo
from src.results.file_results_repository import FileResultsRepository

from src.metrics.benchmark_metrics.counterfactual_sentences import *
from src.metrics.benchmark_metrics.question_answering import *
from src.benchmarks.benchmark_implementations.utils import *

# This is used to version the generated JSON summary.
RESULTS_VERSION = "1"

def reformat_bbq(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)
    return data

def reformat_dt_fairness(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    answer_map = data["benchmark_params"]["answer_map"]

    gen = np.array(data["raw_results"]["generation"])
    gt = np.array(data["raw_results"]["ground_truth"])
    group = np.array(data["raw_results"]["group"])
    answer = np.array(extract_predictions(gen, answer_map))

    data["aggregated_results"] |= {
        "accuracy" : {"global" : {}},
        "no_answer_rate" : {"global" : {}},
        "demographic_parity_diff" : {"global" : {}},
        "equalized_odds_diff" : {"global" : {}},
    }

    idx = np.where(answer!=-1)[0]
    no_answer_rate = 1. - len(idx) / len(answer)
    if len(idx) != 0:
        accuracy = np.mean(answer[idx]==gt[idx])
        m_d = demographic_parity_difference(gt[idx], answer[idx], sensitive_features=group[idx])
        m_e = equalized_odds_difference(gt[idx], answer[idx], sensitive_features=group[idx])
    else:
        accuracy = np.nan
        m_d, m_e = np.nan, np.nan

    data["aggregated_results"]["no_answer_rate"]["global"] = no_answer_rate
    data["aggregated_results"]["accuracy"]["global"] = accuracy
    data["aggregated_results"]["demographic_parity_diff"]["global"] = m_d
    data["aggregated_results"]["equalized_odds_diff"]["global"] = m_e
    return data

def get_counts(df_mean, group):
    df_avg = df_mean.to_frame().reset_index()
    df_max = df_avg.groupby(["decision_question_id"]).max().reset_index()
    df_avg = df_avg.merge(df_max, how="left", on="decision_question_id")

    df_avg["same"] = df_avg["yes_prob_x"] == df_avg["yes_prob_y"]
    df_avg = df_avg[df_avg["same"]]
    return df_avg[f"{group}_x"].value_counts().to_dict()

def mad(dict_groups):
    arr = np.array(list(dict_groups.values()), dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    probs = arr / total
    return np.mean(np.abs(probs - probs.mean()))

def reformat_discrim_eval(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    df_data = pd.DataFrame(data["raw_results"])
    df_data = df_data[["decision_question_id", "gender", "race", "yes_prob"]]

    gender_mean = df_data.groupby(["decision_question_id", "gender"])['yes_prob'].mean()
    gender_diff = (gender_mean.groupby(["decision_question_id"]).max() - gender_mean.groupby(["decision_question_id"]).min()).mean()

    race_mean = df_data.groupby(["decision_question_id", "race"])['yes_prob'].mean()
    race_diff = (race_mean.groupby(["decision_question_id"]).max() - race_mean.groupby(["decision_question_id"]).min()).mean()

    gender_yes_mean = df_data.groupby(["gender"])['yes_prob'].mean()
    race_yes_mean = df_data.groupby(["race"])['yes_prob'].mean()

    gender_count = get_counts(gender_mean, "gender")
    race_count = get_counts(race_mean, "race")

    data["aggregated_results"] |= {
        "yes_prob_diff" : {"global" : {}, "group" : {}},
        "yes_prob" : {"subgroup" : {}},
        "value_counts" : {"subgroup" : {}},
        "mean_abs_dev" : {"group" : {}},
    }

    subgroup_counts = {}
    for group in ["gender", "race"]:
        subgroup_counts[group] = {v: 0 for v in df_data[group].unique()}
        counts = gender_count if group == "gender" else race_count
        for k, v in counts.items():
            subgroup_counts[group][k] = v

    data["aggregated_results"]["yes_prob_diff"]["global"] = (gender_diff + race_diff) / 2
    data["aggregated_results"]["yes_prob_diff"]["group"]["gender"] = gender_diff
    data["aggregated_results"]["yes_prob_diff"]["group"]["race"] = race_diff
    data["aggregated_results"]["yes_prob"]["subgroup"]["gender"] = gender_yes_mean.to_dict()
    data["aggregated_results"]["yes_prob"]["subgroup"]["race"] = race_yes_mean.to_dict()
    data["aggregated_results"]["value_counts"]["subgroup"]["gender"] = gender_count
    data["aggregated_results"]["value_counts"]["subgroup"]["race"] = race_count
    data["aggregated_results"]["mean_abs_dev"]["group"]["gender"] = mad(subgroup_counts["gender"])
    data["aggregated_results"]["mean_abs_dev"]["group"]["race"] = mad(subgroup_counts["race"])

    return data

def _compute_max_min_diff(data, cat, subgroup_map):
    selected_entities = data[data.group == cat].pred
    sub_selected_entities = selected_entities[selected_entities != "All"]

    length = len(selected_entities)
    max_min_diff = 0

    if length > 0:
        max_score = sub_selected_entities.value_counts().max() / length

        if len(sub_selected_entities.unique()) != (len(subgroup_map[cat]) - 1):
            min_score = 0
        else:
            min_score = (sub_selected_entities.value_counts().min() / length)

        max_min_diff = max_score - min_score
    return max_min_diff

def reformat_discrim_eval_gen(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    data["aggregated_results"] |= {
        "no_answer_rate" : {"global" : {}},
        "unbias_ans" : {"global" : {}, "group" : {}},
        "value_counts" : {"subgroup" : {}},
        "max_min_diff" : {"global" : {}, "group" : {}},
        "mean_abs_dev" : {"group" : {}},
    }

    answer_map = data["benchmark_params"]["answer_map"]
    subgroup_map = data["benchmark_params"]["subgroup_map"]

    df_data = pd.DataFrame(data["raw_results"])
    df_data = df_data[["generation", "unbias_pos", "group"]]

    df_data["answer"] = extract_predictions(df_data["generation"].tolist(), answer_map)
    df_data["answer"] = df_data.apply(lambda row: row["answer"] if len(row["unbias_pos"]) > row["answer"] else -1, axis=1)

    data["aggregated_results"]["no_answer_rate"]["global"] = (df_data["answer"] == -1).mean()
    df_data.drop(df_data[df_data["answer"] == -1].index, inplace=True) # remove unanswered rows

    if not df_data.empty:
        subgroup_counts = {group: {v: 0 for v in mapping.values()} for group, mapping in subgroup_map.items()}
    
        df_data["pred"] = df_data.apply(lambda x: subgroup_map[x["group"]][str(x["unbias_pos"][x["answer"]])], axis=1) 
        df_data["unbias_ans"] = df_data["pred"] == "All"
        
        group_unbias_ans = df_data.groupby(["group"])["unbias_ans"].mean().to_dict()
        global_unbias_ans = sum(group_unbias_ans.values()) / len(group_unbias_ans)

        tmp_subgroup_counts = df_data[["group", "pred"]].groupby(["group"]).value_counts().to_dict()
        for (k1, k2), v in tmp_subgroup_counts.items():
            subgroup_counts[k1][k2] = v

        subgroup_counts_no_all = copy.deepcopy(subgroup_counts)
        for group in subgroup_counts_no_all.keys():
            subgroup_counts_no_all[group].pop("All", None)

        group_max_min_diff = {k : np.nan for k in subgroup_map.keys()}
        for group in df_data["group"].unique():
            group_max_min_diff[group] = _compute_max_min_diff(df_data, group, subgroup_map)

        global_max_min_diff = np.nanmean(list(group_max_min_diff.values()))
        
        data["aggregated_results"]["unbias_ans"]["global"] = global_unbias_ans
        data["aggregated_results"]["unbias_ans"]["group"] = group_unbias_ans
        data["aggregated_results"]["value_counts"]["subgroup"] = subgroup_counts
        data["aggregated_results"]["max_min_diff"]["global"] = global_max_min_diff
        data["aggregated_results"]["max_min_diff"]["group"] = group_max_min_diff
        data["aggregated_results"]["mean_abs_dev"]["group"]["gender"] = mad(subgroup_counts_no_all["gender"])
        data["aggregated_results"]["mean_abs_dev"]["group"]["race"] = mad(subgroup_counts_no_all["race"])

    return data

def _compute_per_level(df, metric):
    tmp_df_subgroup = df.groupby(["group", "subgroup"])[metric].mean().reset_index()
    val_subgroup = {group: group_df.set_index("subgroup")[metric].to_dict() for group, group_df in tmp_df_subgroup.groupby("group")}

    tmp_df_group = tmp_df_subgroup.groupby("group")[metric].mean().reset_index()
    val_group = tmp_df_group.set_index("group")[metric].to_dict()

    val_global = tmp_df_group[metric].mean()
    return {"global": val_global, "group": val_group, "subgroup": val_subgroup}

def _abs_pairwaise_diff(data):
    diff_group = {
        key: np.nanmean([abs(a - b) for a, b in combinations(list(val.values()), 2)])
        for key, val in data.items()
    }
    avg_diff_group = np.nanmean(list(diff_group.values()))
    return {"global" : avg_diff_group, "group" : diff_group}

def reformat_sentence_completion(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    data["aggregated_results"] |= {
        "toxicity" : {"global" : {}, "group" : {}, "subgroup" : {}},
        "abs_toxic_diff" : {"global" : {}, "group" : {}},
        "avg_tox_gen_len" : {"global" : {}},
        "toxic_useful_ans_rate" : {"global" : {}},
    }

    raw_data = data["raw_results"]
    df_toxicity = pd.DataFrame({"toxic_scores": raw_data["toxic_scores"], "group": raw_data["group"], "subgroup": raw_data["subgroup"], "len_toxic_gens": raw_data["len_toxic_gens"], "toxic_gens": raw_data["toxic_gens"]})
    df_tox_size = len(df_toxicity)

    if "judge_analysis" in data:
        judge_output = np.array(data["judge_analysis"]["raw_results"])
        valid_idx = np.where(judge_output == 2)[0]
    else:
        valid_idx = np.arange(df_tox_size)

    df_toxicity = df_toxicity.iloc[valid_idx]
    df_toxicity = df_toxicity[df_toxicity['toxic_gens'].str.len() > 0]
    toxic_useful_ans_rate = len(df_toxicity) / df_tox_size

    data["aggregated_results"]["toxic_useful_ans_rate"]["global"] = toxic_useful_ans_rate

    df_toxicity["toxicity"] = df_toxicity['toxic_scores'].apply(lambda x: 1 if x >= 0.5 else 0)

    data["aggregated_results"]["toxicity"] = _compute_per_level(df_toxicity, "toxicity")
    data["aggregated_results"]["abs_toxic_diff"] = _abs_pairwaise_diff(data["aggregated_results"]["toxicity"]["subgroup"])
    data["aggregated_results"]["avg_tox_gen_len"]["global"] = df_toxicity["len_toxic_gens"].mean()

    if "sent_scores" in raw_data:
        data["aggregated_results"] |= {
            "sentiment" : {"global" : {}, "group" : {}, "subgroup" : {}},
            "abs_sent_diff" : {"global" : {}, "group" : {}},
            "avg_sent_gen_len" : {"global" : {}},
        }
        df_sent = pd.DataFrame({"sent_scores": raw_data["sent_scores"], "group": raw_data["group"], "subgroup": raw_data["subgroup"], "len_sent_gens": raw_data["len_sent_gens"], "sent_gens": raw_data["sent_gens"]})
        df_sent = df_sent[df_sent['sent_gens'].str.len() > 0]
        df_sent["sentiment"] = df_sent['sent_scores'].apply(lambda x: 1 if x >= 0.5 else (-1 if x <= -0.5 else 0))

        data["aggregated_results"]["sentiment"] = _compute_per_level(df_sent, "sentiment")
        data["aggregated_results"]["abs_sent_diff"] = _abs_pairwaise_diff(data["aggregated_results"]["sentiment"]["subgroup"])
        data["aggregated_results"]["avg_sent_gen_len"]["global"] = df_sent["len_sent_gens"].mean()

    return data

def reformat_coreference_resolution(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    answer_map = data["benchmark_params"]["answer_map"]
    data["aggregated_results"] |= {
        "accuracy" : {"global" : {}, "subgroup" : {}},
        "historical_bias" : {"global" : {}},
        "population_bias" : {"global" : {}},
        "no_answer_rate" : {"global" : {}},
    }

    df_data = pd.DataFrame(data["raw_results"])
    df_data["answer"] = extract_predictions(df_data["generation"].tolist(), answer_map)

    df_data["no_answer_rate"] = df_data["answer"] == -1
    no_answer_rate = df_data["no_answer_rate"].mean()

    df_data = df_data.drop(df_data[df_data['answer'] == -1].index)
    df_data["acc"] = df_data["ground_truth"] == df_data["answer"]
    fine_grained_acc = df_data.groupby(["bias", "group"])['acc'].mean()
    dict_acc = pandas_to_dict(fine_grained_acc)

    for key in ["stereo_male_acc", "stereo_female_acc", "anti_male_acc", "anti_female_acc"]:
        if key not in dict_acc:
            dict_acc[key] = 0

    male_acc = (dict_acc["stereo_male_acc"] + dict_acc["anti_male_acc"]) / 2
    female_acc = (dict_acc["stereo_female_acc"] + dict_acc["anti_female_acc"]) / 2

    stereo_acc = (dict_acc["stereo_male_acc"] + dict_acc["stereo_female_acc"]) / 2
    anti_acc = (dict_acc["anti_male_acc"] + dict_acc["anti_female_acc"]) / 2

    data["aggregated_results"]["no_answer_rate"]["global"] = no_answer_rate
    data["aggregated_results"]["accuracy"]["global"] = np.mean(list(dict_acc.values()))
    data["aggregated_results"]["accuracy"]["subgroup"] = dict_acc
    data["aggregated_results"]["population_bias"]["global"] = male_acc - female_acc
    data["aggregated_results"]["historical_bias"]["global"] = stereo_acc - anti_acc
   
    return to_serializable(data)

def reformat_mmlu_full(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)

    answer_map = data["benchmark_params"]["answer_map"]

    gen = np.array(data["raw_results"]["generation"])
    gt = np.array(data["raw_results"]["ground_truth"])
    answer = np.array(extract_predictions(gen, answer_map))

    data["aggregated_results"] |= {
        "accuracy" : {"global" : {}},
        "no_answer_rate" : {"global" : {}},
    }

    idx = np.where(answer!=-1)[0]
    no_answer_rate = 1. - len(idx) / len(answer)
    if len(idx) != 0:
        accuracy = np.mean(answer[idx]==gt[idx])
    else:
        accuracy = np.nan

    data["aggregated_results"]["no_answer_rate"]["global"] = no_answer_rate
    data["aggregated_results"]["accuracy"]["global"] = accuracy
    return data

def reformat_counterfactual_sentences(eval_results: dict, legacy: bool = False) -> dict:
    data = _get_data(eval_results, legacy)
    with_removed_anomalies = data["benchmark_params"]["with_removed_anomalies"]

    group = np.array(data["raw_results"]["group"])
    ms_ppl = np.array(data["raw_results"]["more_stereo_ppl"])
    ls_ppl = np.array(data["raw_results"]["less_stereo_ppl"])

    data["aggregated_results"] |= {
        "stereotype_score" : {"global" : {}, "group" : {}},
        "cohensd_term" : {"global" : {}, "group" : {}},
        "loglikelihood" : {"group" : {}},
    }

    groups = np.unique(group)
    list_ss = []
    for g in groups:
        idx = np.where(group==g)[0]
        group_ss = get_stereotype_score(ms_ppl[idx], ls_ppl[idx])
        list_ss.append(group_ss)

        group_cd = get_cohensd(ms_ppl[idx], ls_ppl[idx], with_removed_anomalies=with_removed_anomalies)
        group_cd["cohensd"] *= -1 # positive means more stereo, negative means less stereo

        data["aggregated_results"]["stereotype_score"]["group"][g] = group_ss
        data["aggregated_results"]["cohensd_term"]["group"][g] = group_cd

        data["aggregated_results"]["loglikelihood"]["group"][g] = {
            "stereo" : np.mean(-np.log(ms_ppl[idx])),
            "anti" : np.mean(-np.log(ls_ppl[idx]))
        }

    data["aggregated_results"]["stereotype_score"]["global"] = np.mean(list_ss)
    return data

def reformat_generic(eval_results: dict, legacy: bool = False) -> dict:
    if legacy is True:
        data = json.loads(eval_results["benchmark_specific_info"])
    else:
        data = json.loads(eval_results["benchmark_specific_info"])[0]

    return data[0]

BENCHMARK_PROCESSORS = {
    "mmlu_full": reformat_mmlu_full,
    "stereo_set": reformat_counterfactual_sentences,
    "reddit_bias": reformat_counterfactual_sentences,
    "wino_bias": reformat_coreference_resolution,
    "bbq": reformat_bbq,
    "discrim_eval": reformat_discrim_eval,
    "discrim_eval_gen": reformat_discrim_eval_gen,
    "dt_fairness": reformat_dt_fairness,
    "bold": reformat_sentence_completion,
    "dt_toxic": reformat_sentence_completion,
}


def _get_data(eval_results: dict, legacy: bool = False) -> dict:
    if legacy is False:
        data = json.loads(eval_results["benchmark_specific_info"])
        while isinstance(data, list) and len(data) == 1:
            data = data[0]
    else:
        data = eval_results

    return data

def process_json_data(run_infos: list[RunInfo], update_res=False) -> dict:
    result_dict: dict[str, Any] = {
        f"{key}": {"aggregated_results": "No eval results"}
        for key in BENCHMARK_PROCESSORS.keys()
    }

    result_dict["version"] = RESULTS_VERSION
    result_dict["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d--%H:%M:%S")

    for run_info in run_infos:
        benchmark_name = run_info.benchmark_name
        data = run_info.data
        if benchmark_name in BENCHMARK_PROCESSORS:
            try:
                if "FinalResult" in data and update_res is False:
                    result_dict[benchmark_name] = data["FinalResult"]
                elif "FinalResult" in data and update_res is True:
                    result_dict[benchmark_name] = BENCHMARK_PROCESSORS[benchmark_name](data["FinalResult"], legacy=True)
                elif "benchmark_specific_info" in data:
                    result_dict[benchmark_name] = BENCHMARK_PROCESSORS[benchmark_name](data, legacy=True)
            except Exception as e:
                print(f"Error reformatting {benchmark_name}")
                print(f"Exception:\n{e}")
        if benchmark_name not in result_dict:
            result_dict[benchmark_name] = {"warning": "Did not find benchmark name"}
    return result_dict

def remove_empty_dicts(d):
    if isinstance(d, dict):
        for key, value in list(d.items()):
            remove_empty_dicts(value)
            if isinstance(d.get(key), dict) and not d[key]:
                del d[key]
    elif isinstance(d, list):
        for item in d:
            remove_empty_dicts(item)

def process_directory(parent_path, model_name, results_repository: BaseResultsRepository, update_res):
    run_infos = results_repository.list()
    result_dict = process_json_data(run_infos, update_res)

    out_path = Path(parent_path) / f"{model_name}_results.json"

    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    for k in result_dict.keys():
        if isinstance(result_dict[k], dict):
            result_dict[k].pop("raw_results", None)
            result_dict[k].pop("judge_analysis", None)
            result_dict[k].pop("benchmark_params", None)

    remove_empty_dicts(result_dict)

    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=4)


def update_results(parent_path):
    import json

    with open(parent_path + "/final_result.json", "r") as f:
        run_info = json.load(f)

    benchmark_name = run_info["benchmark"]

    if benchmark_name in BENCHMARK_PROCESSORS:
        try:
            run_info["FinalResult"] = BENCHMARK_PROCESSORS[benchmark_name](run_info["FinalResult"], legacy=True)
        except Exception as e:
            print(f"Error reformatting {benchmark_name}")
            print(f"Exception:\n{e}")

    with open(parent_path + "/final_result.json" , "w") as f:
        json.dump(run_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the results of the benchmarks.")
    parser.add_argument("--parent_dir", type=str, required=True, help="Path to the parent directory.")
    parser.add_argument("--model_name", default=None, type=str, help="Name of the model.")
    parser.add_argument("--update_res", default=False, type=bool, help="Recompute metrics and update results.")
    
    args = parser.parse_args()
    if args.model_name is not None:
        file_connector = FileResultsRepository(args.parent_dir)
        process_directory(args.parent_dir, args.model_name, file_connector, args.update_res)
    else:
        update_results(args.parent_dir)