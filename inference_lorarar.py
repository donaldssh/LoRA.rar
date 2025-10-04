import torch
from lorarar.utils import get_lora_weights
import argparse
import os
from diffusers import StableDiffusionXLPipeline
from lorarar.hypernet import Hypernet


from lorarar.utils import (
    get_lora_weights,
    merge_lora_weights,
    initialize_merged_lora_layer
)


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
        default="loras/subjects/dog8"
    )
    parser.add_argument(
        "--lora2", type=str, help="lora1",
        default="loras/styles/3d_rendering4"
    )
    parser.add_argument(
        "--hypernetwork", type=str, default="models/hypernet.pth"
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
        help="number of image generated"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="qualitative"
    )
    parser.add_argument(
        "--save_weights",
        default=False,
        action="store_true",
        help="Whether to save the merger weights"
    )
    parser.add_argument(
        "--generate_images",
        default=False,
        action="store_true",
        help="Whether to save the images"
    )
    return parser.parse_args()


def main(args):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(0)


    hyper_nn = Hypernet()
    hyper_nn.load_state_dict(torch.load(args.hypernetwork,  weights_only=True))
    hyper_nn.to(device).to(torch.float16)
    args.lora_merge_strategy = "lorarar"

    pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path).to(device).to(dtype=torch.float16)

    res_dir = args.lora1.split("/")[-1].split("-")[-1] + "_" + args.lora2.split("/")[-1].split("-")[-1] 
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, res_dir), exist_ok=True)

    lora_weights = get_lora_weights(args.lora1)
    lora_weights_2 = get_lora_weights(args.lora2)
    unet = pipeline.unet.to(device).to(dtype=torch.float16)

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet

        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        attn_name = ".".join(attn_processor_name.split(".")[:-1])

        attn_module = attn_module.to(device)
        merged_lora_weights_dict = merge_lora_weights(lora_weights, attn_name)
        merged_lora_weights_dict_2 = merge_lora_weights(lora_weights_2, attn_name)
        kwargs = {
            "state_dict": merged_lora_weights_dict,
            "state_dict_2": merged_lora_weights_dict_2,
            "hyper_nn": hyper_nn,
            "args": args,
            "dtype": torch.float16
        }

        attn_module.to_q.set_lora_layer(
            initialize_merged_lora_layer(
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                init_merger_value=1,
                init_merger_value_2=1,
                **kwargs,
            )
        )

        attn_module.to_k.set_lora_layer(
            initialize_merged_lora_layer(
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                init_merger_value=1,
                init_merger_value_2=1,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_merged_lora_layer(
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                init_merger_value=1,
                init_merger_value_2=1,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_merged_lora_layer(
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                init_merger_value=1,
                init_merger_value_2=1,
                **kwargs,
            )
        )

    pipeline.to(device="cuda", dtype=torch.float16)
    unet = pipeline.unet.to(device).to(dtype=torch.float16)

    if args.save_weights:
        attn_processor_list = torch.load("attn_name_list.pth")
        attn_processor_name_list = torch.load("attn_processor_name_list.pth")

        merger_1 = {}
        merger_2 = {}
        weights_distribution_1 = {}
        weights_distribution_2 = {}

        for id_attn, attn_name in enumerate(attn_processor_list):
            attn_module = unet

            weights_distribution_1[id_attn] = {}
            weights_distribution_2[id_attn] = {}
            merger_1[id_attn] = {}
            merger_2[id_attn] = {}

            attn_processor_name =  attn_processor_name_list[id_attn]
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            weights_distribution_1[id_attn] = {
                "q": attn_module.to_q.lora_layer.weight_1.data,
                "v":  attn_module.to_v.lora_layer.weight_1.data,
                "k": attn_module.to_k.lora_layer.weight_1.data,
                "out":  attn_module.to_out[0].lora_layer.weight_1.data
            }
            weights_distribution_2[id_attn] = {
                "q": attn_module.to_q.lora_layer.weight_2.data,
                "v":  attn_module.to_v.lora_layer.weight_2.data,
                "k": attn_module.to_k.lora_layer.weight_2.data,
                "out":  attn_module.to_out[0].lora_layer.weight_2.data
            }

            batch_weight = torch.concatenate((weights_distribution_1[id_attn]["q"].T, weights_distribution_2[id_attn]["q"].T), dim=1)
            merge_coeff = hyper_nn(batch_weight)
            q_1 = merge_coeff[:, 0].data 
            q_2 = merge_coeff[:, 1].data 

            
            batch_weight = torch.concatenate((weights_distribution_1[id_attn]["out"].T, weights_distribution_2[id_attn]["out"].T), dim=1)
            merge_coeff = hyper_nn(batch_weight)
            out_1 = merge_coeff[:, 0] 
            out_2 = merge_coeff[:, 1] 

            merger_1[id_attn] = {
                "q": q_1.detach(),
                "v": attn_module.to_v.lora_layer.merger_1.data.detach(),
                "k": attn_module.to_k.lora_layer.merger_1.data.detach(),
                "out": out_1.detach()
            }

            merger_2[id_attn] = {
                "q": q_2.detach(),
                "v": attn_module.to_v.lora_layer.merger_2.data.detach(),
                "k": attn_module.to_k.lora_layer.merger_2.data.detach(),
                "out": out_2.detach()
            }


        print(f"saving the merger weights")
        torch.save(merger_1, os.path.join(os.path.join(args.save_path, res_dir, "merger_1.pth")))
        torch.save(merger_2, os.path.join(os.path.join(args.save_path, res_dir, "merger_2.pth")))
        torch.save(weights_distribution_1, os.path.join(os.path.join(args.save_path, res_dir, "lora_1.pth")))
        torch.save(weights_distribution_2, os.path.join(os.path.join(args.save_path, res_dir, "lora_2.pth")))

    if args.generate_images:
        for i in range(args.N):
            image = pipeline(prompt=args.prompt, num_inference_steps=args.num_inference_steps, generator=g).images[0]
            image.save(os.path.join(args.save_path, res_dir, f"{i}.jpg"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
