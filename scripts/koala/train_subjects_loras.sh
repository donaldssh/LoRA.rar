#!/bin/bash


declare -A arguments
arguments["backpack"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/backpack --output_dir=loras_koala/subjects/backpack --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["backpack_instance_prompt"]="A תת backpack"
arguments["backpack_validation_prompt"]="A תת backpack on the ground"

arguments["backpack_dog"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/backpack_dog --output_dir=loras_koala/subjects/backpack_dog --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["backpack_dog_instance_prompt"]="A תת backpack"
arguments["backpack_dog_validation_prompt"]="A תת backpack on the ground"

arguments["bear_plushie"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/bear_plushie --output_dir=loras_koala/subjects/bear_plushie --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["bear_plushie_instance_prompt"]="A תת stuffed animal"
arguments["bear_plushie_validation_prompt"]="A תת stuffed animal on the ground"

arguments["berry_bowl"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/berry_bowl --output_dir=loras_koala/subjects/berry_bowl --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["berry_bowl_instance_prompt"]="A תת bowl"
arguments["berry_bowl_validation_prompt"]="A תת bowl on a table"

arguments["can"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/can --output_dir=loras_koala/subjects/can --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["can_instance_prompt"]="A תת can"
arguments["can_validation_prompt"]="A תת can on a floor"

arguments["candle"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/candle --output_dir=loras_koala/subjects/candle --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["candle_instance_prompt"]="A תת candle"
arguments["candle_validation_prompt"]="A תת candle on a table"

arguments["cat"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/cat --output_dir=loras_koala/subjects/cat --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["cat_instance_prompt"]="A תת cat"
arguments["cat_validation_prompt"]="A תת cat on a table"

arguments["cat2"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/cat2 --output_dir=loras_koala/subjects/cat2 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["cat2_instance_prompt"]="A תת cat"
arguments["cat2_validation_prompt"]="A תת cat on a table"

arguments["clock"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/clock --output_dir=loras_koala/subjects/clock --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["clock_instance_prompt"]="A תת clock"
arguments["clock_validation_prompt"]="A תת clock on a table"

arguments["colorful_sneaker"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/colorful_sneaker --output_dir=loras_koala/subjects/colorful_sneaker --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["colorful_sneaker_instance_prompt"]="A תת sneaker"
arguments["colorful_sneaker_validation_prompt"]="A תת sneaker on a table"

arguments["dog"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog --output_dir=loras_koala/subjects/dog --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog_instance_prompt"]="A תת dog"
arguments["dog_validation_prompt"]="A תת dog lying down"

arguments["dog2"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog2 --output_dir=loras_koala/subjects/dog2 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog2_instance_prompt"]="A תת dog"
arguments["dog2_validation_prompt"]="A תת dog lying down"

arguments["dog3"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog3 --output_dir=loras_koala/subjects/dog3 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog3_instance_prompt"]="A תת dog"
arguments["dog3_validation_prompt"]="A תת dog lying down"

arguments["dog5"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog5 --output_dir=loras_koala/subjects/dog5 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog5_instance_prompt"]="A תת dog"
arguments["dog5_validation_prompt"]="A תת dog lying down"

arguments["dog6"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog6 --output_dir=loras_koala/subjects/dog6 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog6_instance_prompt"]="A תת dog"
arguments["dog6_validation_prompt"]="A תת dog in a bucket"

arguments["dog7"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog7 --output_dir=loras_koala/subjects/dog7 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog7_instance_prompt"]="A תת dog"
arguments["dog7_validation_prompt"]="A תת dog lying down"

arguments["dog8"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/dog8 --output_dir=loras_koala/subjects/dog8 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["dog8_instance_prompt"]="A תת dog"
arguments["dog8_validation_prompt"]="A תת dog lying down"

arguments["duck_toy"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/duck_toy --output_dir=loras_koala/subjects/duck_toy --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["duck_toy_instance_prompt"]="A תת toy"
arguments["duck_toy_validation_prompt"]="A תת toy on the table"

arguments["fancy_boot"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/fancy_boot --output_dir=loras_koala/subjects/fancy_boot --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["fancy_boot_instance_prompt"]="A תת boot"
arguments["fancy_boot_validation_prompt"]="A תת boot on the floor"

arguments["grey_sloth_plushie"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/grey_sloth_plushie --output_dir=loras_koala/subjects/grey_sloth_plushie --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["grey_sloth_plushie_instance_prompt"]="A תת stuffed animal"
arguments["grey_sloth_plushie_validation_prompt"]="A תת stuffed animal on the floor"

arguments["monster_toy"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/monster_toy --output_dir=loras_koala/subjects/monster_toy --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["monster_toy_instance_prompt"]="A תת toy"
arguments["monster_toy_validation_prompt"]="A תת toy on the floor"

arguments["pink_sunglasses"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/pink_sunglasses --output_dir=loras_koala/subjects/pink_sunglasses --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["pink_sunglasses_instance_prompt"]="A תת sunglasses"
arguments["pink_sunglasses_validation_prompt"]="A תת sunglasses on the table"

arguments["poop_emoji"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/poop_emoji --output_dir=loras_koala/subjects/poop_emoji --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["poop_emoji_instance_prompt"]="A תת toy"
arguments["poop_emoji_validation_prompt"]="A תת toy on the table"

arguments["rc_car"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/rc_car --output_dir=loras_koala/subjects/rc_car --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["rc_car_instance_prompt"]="A תת toy"
arguments["rc_car_validation_prompt"]="A תת toy on the table"

arguments["red_cartoon"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/red_cartoon --output_dir=loras_koala/subjects/red_cartoon --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["red_cartoon_instance_prompt"]="A תת cartoon"
arguments["red_cartoon_validation_prompt"]="A תת cartoon on the table"

arguments["robot_toy"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/robot_toy --output_dir=loras_koala/subjects/robot_toy --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["robot_toy_instance_prompt"]="A תת toy"
arguments["robot_toy_validation_prompt"]="A תת toy on the couch"

arguments["shiny_sneaker"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/shiny_sneaker --output_dir=loras_koala/subjects/shiny_sneaker --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["shiny_sneaker_instance_prompt"]="A תת sneaker"
arguments["shiny_sneaker_validation_prompt"]="A תת sneaker on the floor"

arguments["teapot"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/teapot --output_dir=loras_koala/subjects/teapot --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["teapot_instance_prompt"]="A תת teapot"
arguments["teapot_validation_prompt"]="A תת teapot on the table"

arguments["vase"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/vase --output_dir=loras_koala/subjects/vase --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["vase_instance_prompt"]="A תת vase"
arguments["vase_validation_prompt"]="A תת vase on the table"

arguments["wolf_plushie"]="--pretrained_model_name_or_path=etri-vilab/koala-lightning-700m --instance_data_dir=datasets/subjects/wolf_plushie --output_dir=loras_koala/subjects/wolf_plushie --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=100 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["wolf_plushie_instance_prompt"]="A תת stuffed animal"
arguments["wolf_plushie_validation_prompt"]="A תת stuffed animal on the floor"

declare -A allocations
allocations[0]="backpack backpack_dog bear_plushie berry_bowl"
allocations[1]="can candle cat cat2"
allocations[2]="clock colorful_sneaker dog dog2"
allocations[3]="dog3 dog5 dog6 dog7"
allocations[4]="duck_toy fancy_boot grey_sloth_plushie"
allocations[5]="monster_toy pink_sunglasses poop_emoji rc_car"
allocations[6]="red_cartoon robot_toy shiny_sneaker teapot"
allocations[7]="vase wolf_plushie dog8"


for gpu in 0 1 2 3 4 5 6 7; do
  (
  for name in ${allocations[${gpu}]}; do
    (
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Start time: $dt"
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch train_single_lora.py ${arguments[${name}]}  --instance_prompt="${arguments[${name}_instance_prompt]}"  --validation_prompt="${arguments[${name}_validation_prompt]}"
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "End time: $dt"
    ) > logs/subject_lora_${name}.txt 2>&1
  done
  ) &
done



