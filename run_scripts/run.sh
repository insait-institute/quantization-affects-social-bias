#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd ..

RUN_NAME="$1"
MODEL_PATH="$2"
CONF_PATH="$3"

TIMESTAMP=$(date "+%Y-%m-%d_%H:%M:%S")

run_job () {
    if [ -n "$2" ]; then
        arg_batch="--batch_size $2"
    else
        arg_batch=""
    fi

    ( python3 run.py "$1" \
            --model_config="$CONF_PATH" \
            --model="$MODEL_PATH" \
            --results_folder="results/runs/$RUN_NAME/$TIMESTAMP" \
            --no-timestamp \
            $arg_batch 
    ) 2> "results/runs/$RUN_NAME/$TIMESTAMP/$(echo $1 | sed 's#/#_#g')_$TIMESTAMP.errors" \
    | tee "results/runs/$RUN_NAME/$TIMESTAMP/$(echo $1 | sed 's#/#_#g')_$TIMESTAMP.log"
}


mkdir -p results/runs/$RUN_NAME/$TIMESTAMP
echo "Capabilities"
echo "└── MMLU"
run_job configs/quest_ans/mmlu_full.yaml

echo "Stereotypes"
echo "└── StereoSet"
run_job configs/counter_sents/stereo_set.yaml 10
echo "└── RedditBias"
run_job configs/counter_sents/reddit_bias.yaml 10
echo "└── WinoBias"
run_job configs/quest_ans/wino_bias.yaml
echo "└── BBQ"
run_job configs/quest_ans/bbq.yaml

echo "Fairness"
echo "└── DiscrimEval"
run_job configs/quest_ans/discrim_eval.yaml
echo "└── DiscrimEvalGen"
run_job configs/quest_ans/discrim_eval_gen.yaml
echo "└── DecodingTrust-Fairness"
run_job configs/quest_ans/dt_fairness.yaml

echo "Toxicity and Sentiment"
echo "└── BOLD"
run_job configs/sents_compl/bold.yaml
echo "└── DecodingTrust-Toxicity"
run_job configs/sents_compl/dt_toxic.yaml

python3 helper_tools/results_processor.py --parent_dir=results/runs/$RUN_NAME/$TIMESTAMP --model_name=$RUN_NAME