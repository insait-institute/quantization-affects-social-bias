import os
import requests
from pathlib import Path
from datasets import load_dataset
import shutil
import tarfile
import time

def save_dataset_from_github(url, dataset_path, folder_name, dataset_filename=None):
    if dataset_filename is None:
        dataset_filename = folder_name

    response = requests.get(url)
    response.raise_for_status()

    final_path = os.path.join(dataset_path, folder_name + '/')
    Path(final_path).mkdir(parents=True, exist_ok=True)

    file_extension = os.path.splitext(url)[1]
    full_filename = dataset_filename + file_extension

    if "tar.gz" in url[-6:]:
        archive_name = url.split("/")[-1]
        archive_path = os.path.join(final_path, archive_name)

        with open(archive_path, 'wb') as f:
            f.write(response.content)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=final_path)

        os.remove(archive_path)
        os.replace(archive_path[:-7] + "/" + dataset_filename, final_path + dataset_filename)
        shutil.rmtree(archive_path[:-7])
    else:
        with open(os.path.join(final_path, full_filename), 'wb') as f:
            f.write(response.content)

def save_dataset_from_hf(hf_path, dataset_path, folder_name, split='train', dataset_filename=None, name=None):
    final_path = os.path.join(dataset_path, folder_name + '/' + split + '/')
    Path(final_path).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(hf_path, split=split, name=name)
    dataset.save_to_disk(final_path)

DATASETS_PATH = "./benchmark_datasets/"

save_dataset_from_github("https://raw.githubusercontent.com/moinnadeem/StereoSet/refs/heads/master/data/dev.json", DATASETS_PATH, "stereo_set")

save_dataset_from_github("https://raw.githubusercontent.com/umanlp/RedditBias/refs/heads/master/data/gender/reddit_comments_gender_male_biased_test_reduced.csv", DATASETS_PATH, "reddit_bias", "gender")
save_dataset_from_github("https://raw.githubusercontent.com/umanlp/RedditBias/refs/heads/master/data/race/reddit_comments_race_white_biased_test_reduced.csv", DATASETS_PATH, "reddit_bias", "race")
save_dataset_from_github("https://raw.githubusercontent.com/umanlp/RedditBias/refs/heads/master/data/religion1/reddit_comments_religion1_christians_biased_test_reduced.csv", DATASETS_PATH, "reddit_bias", "religion_jc")
save_dataset_from_github("https://raw.githubusercontent.com/umanlp/RedditBias/refs/heads/master/data/religion2/reddit_comments_religion2_christians_biased_test_reduced.csv", DATASETS_PATH, "reddit_bias", "religion_mc")
save_dataset_from_github("https://raw.githubusercontent.com/umanlp/RedditBias/refs/heads/master/data/orientation/reddit_comments_orientation_straight_biased_test_reduced.csv", DATASETS_PATH, "reddit_bias", "orientation")

save_dataset_from_github("https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/gender_prompt.json", DATASETS_PATH, "bold", "gender")
save_dataset_from_github("https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/political_ideology_prompt.json", DATASETS_PATH, "bold", "political_ideology")
save_dataset_from_github("https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/profession_prompt.json", DATASETS_PATH, "bold", "profession")
save_dataset_from_github("https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/race_prompt.json", DATASETS_PATH, "bold", "race")
save_dataset_from_github("https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/religious_ideology_prompt.json", DATASETS_PATH, "bold", "religion")

save_dataset_from_github("https://raw.githubusercontent.com/AI-secure/DecodingTrust/refs/heads/main/data/toxicity/user_prompts/toxic.jsonl.gpt4.n%3D25.out1.split.annotated.jsonl.filtered.jsonl", DATASETS_PATH, "decoding_trust_toxicity", "toxic_prompts_gpt4")
time.sleep(5)

save_dataset_from_github("https://raw.githubusercontent.com/nyu-mll/BBQ/refs/heads/main/supplemental/additional_metadata.csv", DATASETS_PATH, "bbq", "additional_metadata")
save_dataset_from_github("https://raw.githubusercontent.com/nyu-mll/BBQ/refs/heads/main/data/Gender_identity.jsonl", DATASETS_PATH, "bbq", "Gender_identity")
save_dataset_from_github("https://raw.githubusercontent.com/nyu-mll/BBQ/refs/heads/main/data/Race_ethnicity.jsonl", DATASETS_PATH, "bbq", "Race_ethnicity")
save_dataset_from_github("https://raw.githubusercontent.com/nyu-mll/BBQ/refs/heads/main/data/Religion.jsonl", DATASETS_PATH, "bbq", "Religion")

save_dataset_from_github("https://raw.githubusercontent.com/uclanlp/corefBias/refs/heads/master/WinoBias/wino/data/anti_stereotyped_type1.txt.test", DATASETS_PATH, "wino_bias", "anti_stereotyped_type1")
save_dataset_from_github("https://raw.githubusercontent.com/uclanlp/corefBias/refs/heads/master/WinoBias/wino/data/anti_stereotyped_type2.txt.test", DATASETS_PATH, "wino_bias", "anti_stereotyped_type2")
save_dataset_from_github("https://raw.githubusercontent.com/uclanlp/corefBias/refs/heads/master/WinoBias/wino/data/pro_stereotyped_type1.txt.test", DATASETS_PATH, "wino_bias", "pro_stereotyped_type1")
save_dataset_from_github("https://raw.githubusercontent.com/uclanlp/corefBias/refs/heads/master/WinoBias/wino/data/pro_stereotyped_type2.txt.test", DATASETS_PATH, "wino_bias", "pro_stereotyped_type2")

save_dataset_from_hf("cais/mmlu", DATASETS_PATH, "mmlu", split='test', name="all")
time.sleep(5)
save_dataset_from_hf("cais/mmlu", DATASETS_PATH, "mmlu", split='dev', name="all")
time.sleep(5)

save_dataset_from_github("https://raw.githubusercontent.com/AI-secure/DecodingTrust/161ae8321ced62f45fcd9ceb412e05b47c603cd4/data/fairness/fairness_data/adult_0_200_test_base_rate_0.0.jsonl", DATASETS_PATH, "decoding_trust_fairness", "adult_data")
save_dataset_from_github("https://raw.githubusercontent.com/AI-secure/DecodingTrust/161ae8321ced62f45fcd9ceb412e05b47c603cd4/data/fairness/fairness_data/sensitive_attr_adult_0_200_test_base_rate_0.0.npy", DATASETS_PATH, "decoding_trust_fairness", "adult_attr")

save_dataset_from_github("https://raw.githubusercontent.com/aisoc-lab/inference-acceleration-bias/refs/heads/main/discrimevalgen/discrim_eval_gen.csv", DATASETS_PATH, "discrim_eval_gen")

save_dataset_from_hf("Anthropic/discrim-eval", DATASETS_PATH, "discrim_eval", split='train', name="explicit")
time.sleep(5)