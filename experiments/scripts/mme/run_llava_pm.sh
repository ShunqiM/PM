#!/bin/bash

model=liuhaotian/llava-v1.5-7b # $root/LLaVA-RLHF-7b-v1.5-224/sft_model

answer_path=${CKPT}/MME/llava/pm
seed=1


base_exp_name=llavav1.5-7b-test-v1
python experiments/eval/model_generate.py \
    --model-path $model \
    --question-file experiments/eval/MME/llava_mme.jsonl \
    --answers-file ${answer_path}/${base_exp_name}/${base_exp_name}-$seed.jsonl \
    --image-folder ${DATA}/MME_Benchmark \
    --conv-mode vicuna_v1 --seed $seed  --return_logits \
    --image_resample  --map_constructor sum_scale --heatmap iterative \
    --iterative_search token_mask --iterative_regularization --exit_threshold 0.3 \
    --normalize_mask norm_enhance --enhance_coe 10 --kernel_size 3 --exit_criteria sum_attention \
    --save_resampled_image