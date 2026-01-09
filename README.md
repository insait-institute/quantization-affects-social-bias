<p align="center">
  <img src="https://github.com/insait-institute/quantization-affects-social-bias/blob/master/docs/static/images/example_gh.svg" width="100%">
</p>

# How Quantization Shapes Bias in LLMs

> #### Federico Marcuzzi, Xuefei Ning, Roy Schwartz, and Iryna Gurevych
>

This repository includes the code and scripts to reproduce the experiments presented in the EACL 2026 paper [How Quantization Shapes Bias in Large Language Models](https://arxiv.org/abs/2508.18088) (paper [website](https://insait-institute.github.io/quantization-affects-social-bias/)). The code can also be used to test social bias on large language models compatible with the HuggingFace library. Our framework is built on top of [COMPL-AI](https://github.com/compl-ai/compl-ai).

## Abstract
This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups.
We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, fairness, toxicity, and sentiment.
We employ both probability- and generated text-based metrics across 13 benchmarks and evaluate models that differ in architecture family and reasoning ability.
Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression.
These trends are generally consistent across demographic categories and subgroups, and model types, although their magnitude depends on the specific setting.
Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice.

---

## Setup

Clone the repository and fetch all submodules:

```bash
git clone https://github.com/insait-institute/quantization-affects-social-bias.git
cd quantization-affects-social-bias
git submodule update --init --recursive
```

[Note] When a path is required to run a script, please provide the absolute path to avoid errors.

[Recommended] Set the HuggingFace home to the model folder at the root of the repository, and export your HF token, which is required to download the benchmarks and models.
```bash
export HF_HOME="./models"
export HF_TOKEN="..."
```

Create the two Conda environments needed to run the Social Bias Evaluation Framework and the quantization library:

* To set up the framework environment, use the ```framework_env.yaml``` file:

  ```bash
  conda env create -f framework_env.yaml
  ```

* [Optional] To set up the quantization library environment, use the ```compression_env.yaml``` file:

  ```bash
  conda env create -f compression/compression_env.yaml
  ```

Download the necessary datasets to run the evaluation:

```bash
conda activate bias_eval
python helper_tools/download_datasets.py
```

[Optional] Download the un-quantized pre-trained models into the `MODELS_DIR` folder:

```bash
conda activate bias_eval
bash helper_tools/download_models.py <MODELS_DIR>
```

[Optional] Quantize the model as described in the article. After quantization, each model will be saved in a dedicated folder within `MODELS_DIR` (note: `MODELS_DIR` is the folder containing the root folder of the models to be quantized).

```bash
conda activate compress
cd compress_models <MODELS_DIR>
bash compress_models.sh
```

## Test Framework

* [Fast] To test the installation of the Social Bias Evaluation Framework on a dummy model, run the following command:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_test.sh
  ```

* [Slow] To test the framework on the LLM model saved in the `MODEL_PATH` folder, run the following script. The script will load the LLM and run the evaluation on a subset of each evaluation benchmark.

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_debug.sh <MODEL_PATH>
  ```

## Run Full Evaluation

* To fully evaluate a model, use the following commands, where `MODEL_PATH` is the model folder and `CONFIG_PATH` is the model configuration file stored in `./configs/models/`:

```bash
  conda activate bias_eval
  cd run_scripts
  bash run.sh <MODEL_PATH> <CONFIG_PATH>
  ```

* To reproduce the evaluation performed in the article, run the following:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_all.sh <MODELS_DIR>
  ```

## Run LLM-as-a-judge Evaluation

* To run the LLM-as-a-judge evaluation on toxic continuations, use the following, where `BENCH_RESULTS_DIR` is the folder containing the benchmark results (e.g., `results/runs/Qwen2.5-14B-Instruct/1984-04-30_00:00:00`), `BENCH_NAME` can be `bold` or `dt_toxic`, `MODEL_NAME` is the name of the model whose results you want to evaluate, and [optional] `JUDGE_PATH` is the path to the judge model.

```bash
  conda activate bias_eval
  cd run_scripts
  bash run_judge.sh <BENCH_RESULTS_DIR> <BENCH_NAME> <MODEL_NAME> <JUDGE_PATH>
  ```

* To reproduce the  LLM-as-a-judge evaluation performed in the article, run the following:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_judge_all.sh
  ```

## Compute Model Size

To compute the size of the un-quantized model as well as the non-fake-quantized models reported in the article, run the following commands:

 ```bash
conda activate bias_eval
python helper_tools/compute_model_size.py 
```

---

## Citation
```
@article{marcuzzi2026quantizationshapesllmsbias,
    author    = {Federico Marcuzzi and Xuefei Ning and Roy Schwartz and Iryna Gurevych},
    title     = {How Quantization Shapes Bias in Large Language Models},
    booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics, {EACL} 2026 - Volume 1: Long Papers, Rabat, Morocco, March 24-29, 2026},
    pages     = {To appear},
    publisher = {Association for Computational Linguistics},
    year      = {2026},
    url       = {https://arxiv.org/abs/2508.18088},
    note      = {Accepted to the main conference (EACL 2026)}
}
```