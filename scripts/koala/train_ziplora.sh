#!/bin/bash

declare -A hashmap
hashmap["dog8"]="dog"
hashmap["cat2"]="cat"
hashmap["wolf_plushie"]="stuffed animal"
hashmap["teapot"]="teapot"
hashmap["can"]="can"

hashmap["3d_rendering4"]="3d rendering"
hashmap["oil_painting2"]="oil painting"
hashmap["watercolor_painting3"]="watercolor painting"
hashmap["flat_cartoon_illustration"]="flat cartoon illustration"
hashmap["glowing"]="glowing"

test_subjects=("dog8" "cat2" "wolf_plushie" "teapot" "can")
test_styles=("3d_rendering4" "oil_painting2" "watercolor_painting3" "flat_cartoon_illustration" "glowing")

job_id=0

for subject in "${test_subjects[@]}"; do
  for style in "${test_styles[@]}"; do
    prompt="a תת ${hashmap[$subject]} in ${hashmap[$style]} style"

    echo "Submitting job $job_id: $subject + $style"

    CUDA_VISIBLE_DEVICES=0 accelerate launch train_lorarar.py \
    --pretrained_model_name_or_path=etri-vilab/koala-lightning-700m \
    --output_dir=ziplora_models/_koala/${subject}_${style} \
    --lora_name_or_path=loras_koala/subjects/$subject \
    --instance_prompt="\"$(jq -r .subjects.\"$subject\".instance_prompt captions/all.json)\"" \
    --instance_data_dir=datasets/subjects/$subject \
    --lora_name_or_path_2=loras_koala/styles/$style \
    --instance_prompt_2="\"$(jq -r .styles.\"$style\".instance_prompt captions/all.json)\"" \
    --instance_data_dir_2=datasets/styles/$style \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=0.01 \
    --similarity_lambda=0.01 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps=100 \
    --validation_prompt="$prompt" \
    --validation_epochs=20 \
    --seed=0 \
    --mixed_precision=fp16 \
    --report_to=tensorboard \
    --gradient_checkpointing \
    --use_8bit_adam \
    --lora_merge_strategy ziplora \
    > logs/ziplora_${subject}_${style}.txt 2>&1 &
   
    ((job_id++))
    echo ""
  done
done
