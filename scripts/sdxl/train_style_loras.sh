#!/bin/bash


declare -A arguments

arguments["3d_rendering"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/3d_rendering --output_dir=loras/styles/3d_rendering --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["3d_rendering_instance_prompt"]="Slice of watermelon and clouds in the background in 3d rendering style"
arguments["3d_rendering_validation_prompt"]="A penguin in 3d rendering style"

arguments["3d_rendering2"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/3d_rendering2 --output_dir=loras/styles/3d_rendering2 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["3d_rendering2_instance_prompt"]="A yellow and brown duck cartoon character in 3d rendering style"
arguments["3d_rendering2_validation_prompt"]="A penguin in 3d rendering style"

arguments["3d_rendering3"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/3d_rendering3 --output_dir=loras/styles/3d_rendering3 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["3d_rendering3_instance_prompt"]="A house in 3d rendering style"
arguments["3d_rendering3_validation_prompt"]="A penguin in 3d rendering style"

arguments["3d_rendering4"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/3d_rendering4 --output_dir=loras/styles/3d_rendering4 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["3d_rendering4_instance_prompt"]="A women in 3d rendering style"
arguments["3d_rendering4_validation_prompt"]="A penguin in 3d rendering style"

arguments["abstract_rainbow_colored_flowing_smoke_wave_design"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/abstract_rainbow_colored_flowing_smoke_wave_design --output_dir=loras/styles/abstract_rainbow_colored_flowing_smoke_wave_design --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["abstract_rainbow_colored_flowing_smoke_wave_design_instance_prompt"]="A wave in abstract rainbow colored flowing smoke wave design"
arguments["abstract_rainbow_colored_flowing_smoke_wave_design_validation_prompt"]="A dinosaur in abstract rainbow colored flowing smoke wave design"

arguments["black_statue"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/black_statue --output_dir=loras/styles/black_statue --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["black_statue_instance_prompt"]="A gargoyle in black statue"
arguments["black_statue_validation_prompt"]="A penguin in black statue"

arguments["cartoon_line_drawing"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/cartoon_line_drawing --output_dir=loras/styles/cartoon_line_drawing --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["cartoon_line_drawing_instance_prompt"]='A person drowning into the phone in cartoon line drawing style'
arguments["cartoon_line_drawing_validation_prompt"]="A phone in cartoon line drawing style"

arguments["kid_crayon_drawing"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/kid_crayon_drawing --output_dir=loras/styles/kid_crayon_drawing --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["kid_crayon_drawing_instance_prompt"]='A bear in kid crayon drawing style'
arguments["kid_crayon_drawing_validation_prompt"]='A penguin in kid crayon drawing style'

arguments["flat_cartoon_illustration"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/flat_cartoon_illustration --output_dir=loras/styles/flat_cartoon_illustration --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["flat_cartoon_illustration_instance_prompt"]='A woman working on a laptop in flat cartoon illustration style'
arguments["flat_cartoon_illustration_validation_prompt"]='A dog in flat cartoon illustration style'

arguments["flat_cartoon_illustration2"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/flat_cartoon_illustration2 --output_dir=loras/styles/flat_cartoon_illustration2 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["flat_cartoon_illustration2_instance_prompt"]='A women walking a dog in flat cartoon illustration style'
arguments["flat_cartoon_illustration2_validation_prompt"]='A dog in flat cartoon illustration style'

arguments["glowing"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/glowing --output_dir=loras/styles/glowing  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["glowing_instance_prompt"]='A mushroom in glowing style'
arguments["glowing_validation_prompt"]='A penguin in glowing style'

arguments["glowing_3d_rendering"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/glowing_3d_rendering --output_dir=loras/styles/glowing_3d_rendering  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["glowing_3d_rendering_instance_prompt"]='A thumbs up in glowing 3d rendering style'
arguments["glowing_3d_rendering_validation_prompt"]='A lighter in glowing 3d rendering style'

arguments["line_drawing"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/line_drawing --output_dir=loras/styles/line_drawing  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["line_drawing_instance_prompt"]='A village in line drawing style'
arguments["line_drawing_validation_prompt"]='A forest in line drawing style'

arguments["melting_golden_rendering"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/melting_golden_rendering --output_dir=loras/styles/melting_golden_rendering  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["melting_golden_rendering_instance_prompt"]='A flower in melting golden 3d rendering style'
arguments["melting_golden_rendering_validation_prompt"]='A pizza in melting golden 3d rendering style'

arguments["oil_painting"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/oil_painting --output_dir=loras/styles/oil_painting  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["oil_painting_instance_prompt"]='A village in oil painting style'
arguments["oil_painting_validation_prompt"]='A forest in oil painting style'

arguments["oil_painting2"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/oil_painting2 --output_dir=loras/styles/oil_painting2  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["oil_painting2_instance_prompt"]='A portrait of a person in oil painting style'
arguments["oil_painting2_validation_prompt"]='A portrait of a dog in oil painting style'

arguments["oil_painting3"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/oil_painting3 --output_dir=loras/styles/oil_painting3  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["oil_painting3_instance_prompt"]='A portrait of a person wearing a hat in oil painting style'
arguments["oil_painting3_validation_prompt"]='A portrait of a dog in oil painting style'

arguments["sticker"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/sticker --output_dir=loras/styles/sticker  --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000  --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["sticker_instance_prompt"]='A Christmas tree in sticker style'
arguments["sticker_validation_prompt"]='A penguin in sticker style'

arguments["watercolor_painting"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting --output_dir=loras/styles/watercolor_painting --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting_instance_prompt"]='Flowers in watercolor painting style'
arguments["watercolor_painting_validation_prompt"]='A dog in watercolor painting style'

arguments["watercolor_painting2"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting2 --output_dir=loras/styles/watercolor_painting2 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting2_instance_prompt"]='Trees during daytime in watercolor painting style'
arguments["watercolor_painting2_validation_prompt"]='A dog in watercolor painting style'

arguments["watercolor_painting3"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting3 --output_dir=loras/styles/watercolor_painting3 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting3_instance_prompt"]='A girl in blue dress holding a red rose in watercolor painting style'
arguments["watercolor_painting3_validation_prompt"]='A dog in watercolor painting style'

arguments["watercolor_painting4"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting4 --output_dir=loras/styles/watercolor_painting4 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting4_instance_prompt"]='A orange and white fox standing on boulder in watercolor painting style'
arguments["watercolor_painting4_validation_prompt"]='A dog in watercolor painting style'

arguments["watercolor_painting5"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting5 --output_dir=loras/styles/watercolor_painting5 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting5_instance_prompt"]='A bay in watercolor painting style'
arguments["watercolor_painting5_validation_prompt"]='A mountain in watercolor painting style'

arguments["watercolor_painting6"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting6 --output_dir=loras/styles/watercolor_painting6 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting6_instance_prompt"]="A cat in watercolor painting style"
arguments["watercolor_painting6_validation_prompt"]="A man in watercolor painting style"

arguments["watercolor_painting7"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/watercolor_painting7 --output_dir=loras/styles/watercolor_painting7 --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["watercolor_painting7_instance_prompt"]='A house in watercolor painting style'
arguments["watercolor_painting7_validation_prompt"]='A dog in watercolor painting style'

arguments["wooden_sculpture"]="--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --instance_data_dir=datasets/styles/wooden_sculpture --output_dir=loras/styles/wooden_sculpture --rank=64 --resolution=1024 --train_batch_size=1 --learning_rate=5e-5 --report_to=tensorboard --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=1000 --validation_epochs=50 --seed=0 --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam"
arguments["wooden_sculpture_instance_prompt"]='A Viking face with beard in wooden sculpture'
arguments["wooden_sculpture_validation_prompt"]='A dog face in wooden sculpture'


declare -A allocations
allocations[0]="3d_rendering 3d_rendering2 3d_rendering3 3d_rendering4"
allocations[1]="abstract_rainbow_colored_flowing_smoke_wave_design black_statue cartoon_line_drawing"
allocations[2]="flat_cartoon_illustration flat_cartoon_illustration2 glowing glowing_3d_rendering"
allocations[3]="line_drawing melting_golden_rendering oil_painting oil_painting2"
allocations[4]="oil_paiting3 sticker watercolor_painting2 watercolor_painting"
allocations[5]="watercolor_painting3 watercolor_painting4 watercolor_painting5"
allocations[6]="watercolor_painting7 wooden_sculture watercolor_painting6 kid_crayon_drawing"


for gpu in 0 1 2 3 4 5 6 7; do
  (
  for name in ${allocations[${gpu}]}; do
    (
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Start time: $dt"
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch train_single_lora.py ${arguments[${name}]}  --instance_prompt="${arguments[${name}_instance_prompt]}"  --validation_prompt="${arguments[${name}_validation_prompt]}"
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "End time: $dt"
    ) > logs/style_lora_${name}.txt 2>&1
  done
  ) &
done



