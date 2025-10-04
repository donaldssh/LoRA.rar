import torch
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline
import argparse
import os
import copy
from peft import get_peft_model, PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pretrained model path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lora1", type=str, help="lora1",
        default="single_loras/subjects/lora-sdxl-dog8"
    )
    parser.add_argument(
        "--lora2", type=str, help="lora1",
        default="single_loras/styles/lora-sdxl-3d_rendering4"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a תת dog in 3d rendering style"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="number of images generated"
    )
    parser.add_argument(
        "--merge_strategy",
        type=str,
        default="linear",
        help="[svd, linear, cat, ties, ties_svd, dare_ties, dare_linear, dare_ties_svd, dare_linear_svd, magnitude_prune, magnitude_prune_svd]"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="uniform",
        help="merging weights in the form of e.g. 0.3,0.3,0.4 if not uniform"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="density parameter used for some of the merging methods"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="generated_images/peft"
    )
    return parser.parse_args()


args = parse_args()

res_dir = args.lora1.split("/")[-1].split("-")[-1] + "_" + args.lora2.split("/")[-1].split("-")[-1] 
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, res_dir), exist_ok=True)

g = torch.Generator(device="cuda")
g.manual_seed(0)

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

sdxl_unet = copy.deepcopy(unet)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    args.pretrained_model_name_or_path, variant="fp16", torch_dtype=torch.float16, unet=unet).to("cuda")

pipeline.load_lora_weights(
    os.path.join(args.lora1, "pytorch_lora_weights.safetensors"),
    adapter_name="subject"
)

subject_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["subject"],
    adapter_name="subject"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
subject_peft_model.load_state_dict(original_state_dict, strict=True)
subject_peft_model.save_pretrained("subject_peft_model")

pipeline.delete_adapters("subject")
sdxl_unet.delete_adapters("subject")

pipeline.load_lora_weights(
    os.path.join(args.lora2, "pytorch_lora_weights.safetensors"),
    adapter_name="style"
)
pipeline.set_adapters(adapter_names="style")

style_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["style"],
    adapter_name="style"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
style_peft_model.load_state_dict(original_state_dict, strict=True)
style_peft_model.save_pretrained("style_peft_model")

base_unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

model = PeftModel.from_pretrained(base_unet, "subject_peft_model",
                                  use_safetensors=True, subfolder="subject", adapter_name="subject")
model.load_adapter("style_peft_model", use_safetensors=True,
                   subfolder="style", adapter_name="style")

# perform the merge - create a new weighted adapter
if args.weights == "uniform":
    weights = [0.5, 0.5]
else:
    # weights is a str in the format e.g. 0.3,0.3,0.4
    weights = [float(x) for x in args.weights.split(",")]

model.add_weighted_adapter(
    adapters=["subject", "style"],
    weights=weights,
    density=args.density,
    combination_type=args.merge_strategy,
    adapter_name="merge"
)
model.set_adapters("merge")

model = model.to(dtype=torch.float16, device="cuda")

pipeline = StableDiffusionXLPipeline.from_pretrained(
    args.pretrained_model_name_or_path, variant="fp16", torch_dtype=torch.float16, unet=model).to("cuda")

for i in range(args.N):
    image = pipeline(prompt=args.prompt, num_inference_steps=args.num_inference_steps, generator=g).images[0]
    image.save(os.path.join(args.save_path, res_dir, f"{i}.jpg"))

