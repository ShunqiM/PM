datasets=("aokvqa" "gqa" "coco")
types=("adversarial" "popular" "random")

model_path="liuhaotian/llava-v1.5-7b"

base_exp_name=llavav1.5-7b-test-v1

for dataset_name in "${datasets[@]}"; do
  # Determine the appropriate image folder
  if [[ $dataset_name == "coco" || $dataset_name == "aokvqa" ]]; then
    image_folder="${DATA}/POPE/coco/val2014"
  else
    image_folder="${DATA}/POPE/gqa/images"
  fi

  # Iterate through each type
  for type in "${types[@]}"; do
    for seed in {55..55}; do  # Adjust this range as per the number of splits

        python experiments/eval/model_generate.py \
            --model-path ${model_path} \
            --question-file ./experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
            --image-folder ${image_folder} \
            --answers-file ${CKPT}/POPE/pm/${save_dir}/llava_${dataset_name}_pope_${type}_seed${seed}.jsonl \
            --conv-mode vicuna_v1 --seed $seed  --return_logits \
            --image_resample  --map_constructor sum_scale --heatmap iterative \
            --iterative_search token_mask --iterative_regularization --exit_threshold 0.4 \
            --normalize_mask norm_enhance --enhance_coe 15 --kernel_size 5 --exit_criteria sum_attention
    done
  done
done
