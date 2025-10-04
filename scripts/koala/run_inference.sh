#!/bin/bash

declare -a test_subjects=("dog8" "cat2" "wolf_plushie" "teapot" "can")
declare -a test_styles=("3d_rendering4" "oil_painting2" "watercolor_painting3" "flat_cartoon_illustration" "glowing")

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

export HYPERNET_PATH="models/koala_700m_hypernet.pth"
export LORAS_HOME="loras_koala"
export SAVE_PATH="generated_images_koala/lorarar"

for subject in "${test_subjects[@]}" 
do
        for style in "${test_styles[@]}" 
        do
        prompt="a תת ${hashmap[$subject]} in ${hashmap[$style]} style"

        CUDA_VISIBLE_DEVICES=0 python inference_lorarar.py \
        --pretrained_model_name_or_path="etri-vilab/koala-lightning-700m" \
        --N 10 --lora1 $LORAS_HOME/subjects/$subject \
        --lora2  $LORAS_HOME/styles/$style  --prompt="'${prompt}'" \
        --hypernetwork $HYPERNET_PATH \
        --save_path $SAVE_PATH \
        --generate_images  > logs/inference_${subject}_${style}.txt 2>&1

        done
done

