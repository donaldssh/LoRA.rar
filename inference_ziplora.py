import torch
from ziplora_pytorch.utils import insert_ziplora_to_unet
import argparse
import os
from diffusers import StableDiffusionXLPipeline


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
        "--ziplora_name_or_path", type=str, help="ziplora path", default="ziplora_models/dog8_3d_rendering4"
    )
    parser.add_argument(
        "--save_path", type=str, required=False, help="save path", default="generated_images/ziplora"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='a תת dog in 3d rendering style'
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="number of image generated"
    )
    return parser.parse_args()

args = parse_args()

g = torch.Generator(device="cuda")
g.manual_seed(0)

pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path)

res_dir = args.ziplora_name_or_path.split('/')[-1]
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, res_dir), exist_ok=True)

pipeline.unet = insert_ziplora_to_unet(pipeline.unet, args.ziplora_name_or_path)
pipeline.to(device="cuda", dtype=torch.float16)
for i in range(args.N):
    image = pipeline(prompt=args.prompt, num_inference_steps=args.num_inference_steps, generator=g).images[0]
    image.save(os.path.join(args.save_path, res_dir, f"{i}.jpg"))
