#!/bin/bash


export MODEL_NAME="etri-vilab/koala-lightning-700m"


CUDA_VISIBLE_DEVICES=0 accelerate launch train_lorarar.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="models/lorarar" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=0.01 \
  --similarity_lambda=0.01 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=100 \
  --validation_epochs=7777777 \
  --seed="0" \
  --mixed_precision="fp16" \
  --report_to="tensorboard" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --subject_loras_path loras_koala/subjects \
  --style_loras_path loras_koala/styles \
  --home_instance_data_dir_hyper datasets \
  --caption_path captions/train_split.json \
  --lora_merge_strategy lorarar \
  --log_merger_weights \
  --debug  > logs/lorarar_training.txt 2>&1
  