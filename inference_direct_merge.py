import torch
from diffusers import StableDiffusionXLPipeline
import argparse
import os


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
        default=""
    )
    parser.add_argument(
        "--lora1_w", type=float, help="lora1_w",
        default=1
    )
    parser.add_argument(
        "--lora2", type=str, help="lora1",
        default=""
    )
    parser.add_argument(
        "--lora2_w", type=float, help="lora2_w",
        default=1
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=""
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="number of image generated"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="generated_images/direct_merge"
    )
    return parser.parse_args()

args = parse_args()

res_dir = args.lora1.split("/")[-1].split("-")[-1] + f"_{args.lora1_w}_" + args.lora2.split("/")[-1].split("-")[-1] + f"_{args.lora2_w}"
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, res_dir), exist_ok=True)


g = torch.Generator(device="cuda")
g.manual_seed(0)

pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path)

lora1_model = pipeline.load_lora_weights(
    os.path.join(args.lora1, "pytorch_lora_weights.safetensors"),
    adapter_name="subject"
    )
lora2_model = pipeline.load_lora_weights(
    os.path.join(args.lora2, "pytorch_lora_weights.safetensors"),
    adapter_name="style"
    )

pipeline.set_adapters(["subject", "style"], adapter_weights=[args.lora1_w, args.lora2_w])

pipeline.to(device="cuda", dtype=torch.float16)

for i in range(args.N):
    image = pipeline(prompt=args.prompt, num_inference_steps=args.num_inference_steps, generator=g).images[0]
    image.save(os.path.join(args.save_path, res_dir, f"{i}.jpg"))

