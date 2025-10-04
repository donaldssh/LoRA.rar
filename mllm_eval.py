import argparse
import torch
from PIL import Image
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image
import copy
import torch

import warnings
import os
import json


def get_model(args):
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-critic-7b"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    model.eval()

    return model, tokenizer, image_processor, device


def get_image(image_path):
    image = Image.open(image_path)

    # check the size of the image and reshape it so that the longer side is args.image_size
    width, height = image.size
    if width >= height and width > args.image_size:
        ratio = args.image_size / width
        new_width = args.image_size
        new_height = int(height * ratio)
        new_size = (new_width, new_height)
        image = image.resize(new_size)
    elif height >= width and height > args.image_size:
        ratio = args.image_size / height
        new_height = args.image_size
        new_width = int(width * ratio)
        new_size = (new_width, new_height)
        image = image.resize(new_size)

    return image


def compare_reference(model, tokenizer, image_processor, test_image, test_image_tensor, device, reference_image_path, eval_type='subject', use_score=False, short_answer=False):
    with torch.no_grad():
        support_image = get_image(reference_image_path)
        support_image_tensor = process_images([support_image], image_processor, model.config)[
            0].to(dtype=torch.float16, device=device)

        if use_score:
            if eval_type == 'subject':

                question = f"""Your task is to score to what extent the test image shows the same subject as the support image.\n

Support image:\n
{DEFAULT_IMAGE_TOKEN}\n
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
Pay attention to details of the subject, it should for example have the same colour. However, the general style of the image may be different. \n

Scoring guidelines:\n
Score between 0 to 25 if the subject in the test image is completely different from the subject in the support image, for example different animal species or category.\n
Score between 26 to 50 if the subject in the test image is slighty similar to the subject in the support image, for example has different colour.\n
Score between 51 to 75 if the subject in the test image is quite similar to the subject in the support image.\n
Score between 76 to 100 if the subject in the test image is exactly the same as the subject in the support image.\n

How well does the subject in the test image match the subject in the support image? Answer with a number only and no explanation, with the number between 0 and 100. Example answer: 50""".strip()

            elif eval_type == 'style':
                style = reference_image_path.split('/')[-2]
                if style[-1] in '0123456789':
                    style = style[:-1]
                style = style.replace('_', ' ')

                question = f"""Your task is to score to what extent the test image shows the subject in {style} style. An example image in the {style} style is provided.\n

Illustration of the {style} style - ignore the details of the subject:\n
{DEFAULT_IMAGE_TOKEN}\n   
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
The example image shows an illustration of the {style} style and the details of the subject are expected to be different. Do not check similarity with the subject.

Do not be very strict with the style, some differences are acceptable.

Scoring guidelines:\n
Score between 0 to 25 if the style of the test image is completely different from the {style} style.\n
Score between 26 to 50 if the style of the test image is slightly similar to the {style} style.\n
Score between 51 to 75 if the style of the test image is quite similar to the {style} style.\n
Score between 76 to 100 if the style of the test image is exactly the same as the {style} style.\n
            
How well does the style used in the test image match the {style} style? Answer with a number only and no explanation, with the number between 0 and 100. Example answer: 50""".strip()

            else:
                raise ValueError(f"Unknown eval type: {eval_type}")
        elif short_answer:
            if eval_type == 'subject':
                question = f"""Your task is to identify if the test image shows the same subject as the support image.\n
Support image:\n
{DEFAULT_IMAGE_TOKEN}\n
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
Pay attention to details of the subject, it should for example have the same colour. However, the general style of the image may be different. \n
Does the test image show the same subject as the support image? Answer with Yes or No only.""".strip()
            elif eval_type == 'style':
                style = reference_image_path.split('/')[-2]
                if style[-1] in '0123456789':
                    style = style[:-1]
                style = style.replace('_', ' ')
                question = f"""Your task is to identify if the test image shows the subject in {style} style. An example image in the {style} style is provided.\n
Example image in the {style} style:\n
{DEFAULT_IMAGE_TOKEN}\n   
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
The example image shows an illustration of the {style} style and the details of the subject are expected to be different. Do not check similarity with the subject.
Is the test image in the {style} style?
Answer with Yes or No only.""".strip()
            else:
                raise ValueError(f"Unknown eval type: {eval_type}")
        else:
            if eval_type == 'subject':
                question = f"""Your task is to identify if the test image shows the same subject as the support image.\n
Support image:\n
{DEFAULT_IMAGE_TOKEN}\n
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
Pay attention to details of the subject, it should for example have the same colour. However, the general style of the image may be different. \n
Does the test image show the same subject as the support image? Answer with Yes or No, and a short explanation.""".strip()
            elif eval_type == 'style':
                style = reference_image_path.split('/')[-2]
                if style[-1] in '0123456789':
                    style = style[:-1]
                style = style.replace('_', ' ')
                question = f"""Your task is to identify if the test image shows the subject in {style} style. An example image in the {style} style is provided.\n
Example image in the {style} style:\n
{DEFAULT_IMAGE_TOKEN}\n   
Test image:\n
{DEFAULT_IMAGE_TOKEN}\n
The example image shows an illustration of the {style} style and the details of the subject are expected to be different. Do not check similarity with the subject.
Is the test image in the {style} style?
Answer with Yes or No, and a short explanation.""".strip()
            else:
                raise ValueError(f"Unknown eval type: {eval_type}")

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [support_image.size, test_image.size]

        cont = model.generate(
            input_ids,
            images=[support_image_tensor, test_image_tensor],
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

        torch.cuda.empty_cache()

        return text_outputs[0]


def compare_references(model, tokenizer, image_processor, test_image, device, reference_dir, eval_type='subject', use_score=False, short_answer=True):
    test_image_tensor = process_images([test_image], image_processor, model.config)[
        0].to(dtype=torch.float16, device=device)

    accuracies = []
    for support_image_path in os.listdir(reference_dir):
        text_output = compare_reference(model, tokenizer, image_processor, test_image, test_image_tensor, device, os.path.join(
            reference_dir, support_image_path), eval_type, use_score, short_answer)

        if 'yes' in text_output.strip().lower():
            accuracies.append(1)
        else:
            accuracies.append(0)

    return accuracies


def evaluate_pair(model, tokenizer, image_processor, device, test_image_path, reference_image_path, eval_type='subject', use_score=False, short_answer=False):
    test_image = get_image(test_image_path)
    test_image_tensor = process_images([test_image], image_processor, model.config)[
        0].to(dtype=torch.float16, device=device)

    text_output = compare_reference(model, tokenizer, image_processor, test_image,
                                    test_image_tensor, device, reference_image_path, eval_type, use_score, short_answer)
    print(text_output)


def evaluate_image(model, tokenizer, image_processor, test_image_path, device, reference_dir, subject, style):
    test_image = get_image(test_image_path)
    subject_accuracies = compare_references(model, tokenizer, image_processor, test_image, device, os.path.join(
        reference_dir, "subjects", subject), eval_type='subject', use_score=False, short_answer=True)
    style_accuracies = compare_references(model, tokenizer, image_processor, test_image, device, os.path.join(
        reference_dir, "styles", style), eval_type='style', use_score=False, short_answer=True)

    # do a majority vote for both subject and style
    correct_subject = int(sum(subject_accuracies) /
                          len(subject_accuracies) > 0.5)
    correct_style = int(sum(style_accuracies) / len(style_accuracies) > 0.5)

    return (correct_subject, correct_style)


def main(args):
    model, tokenizer, image_processor, device = get_model(args)
    all_subjects = ["can", "cat2", "dog8", "teapot", "wolf_plushie"]
    all_styles = ["3d_rendering4", "flat_cartoon_illustration",
                  "glowing", "oil_painting2", "watercolor_painting3"]

    results = {"generated_imgs_dir": args.generated_imgs_dir, "results": []}
    for scenario_dir in os.listdir(args.generated_imgs_dir):
        print(scenario_dir)
        subject = None
        for current_subject in all_subjects:
            if current_subject in scenario_dir:
                subject = current_subject
        style = None
        for current_style in all_styles:
            if current_style in scenario_dir:
                style = current_style

        if not subject or not style:
            print(f"Skipping scenario {scenario_dir}")
            continue

        correct_subjects = []
        correct_styles = []
        generated_image_paths = []

        generated_image_paths_subset = os.listdir(os.path.join(args.generated_imgs_dir, scenario_dir))[:args.num_eval_images]
        for generated_image_path in generated_image_paths_subset:
            test_image_path = os.path.join(
                args.generated_imgs_dir, scenario_dir, generated_image_path)
            correct_subject, correct_style = evaluate_image(
                model, tokenizer, image_processor, test_image_path, device, args.reference_dir, subject, style)
            print(generated_image_path, correct_subject, correct_style)
            correct_subjects.append(correct_subject)
            correct_styles.append(correct_style)
            generated_image_paths.append(generated_image_path)

        results_dict = {"scenario_dir": scenario_dir,
                        "generated_image_paths": generated_image_paths,
                        "subject": subject,
                        "style": style,
                        "correct_subjects": correct_subjects,
                        "correct_styles": correct_styles,
                        "avg_correct_subjects": np.mean(correct_subjects),
                        "avg_correct_styles": np.mean(correct_styles)}
        results["results"].append(results_dict)

        print(f'Scenario: {scenario_dir}')
        print(f'Accurately generated subject: {np.mean(correct_subjects)}')
        print(f'Accurately generated style: {np.mean(correct_styles)}')

    # make directory mllm_evaluations if does not exist
    if not os.path.exists("mllm_evaluations"):
        os.makedirs("mllm_evaluations")
    with open(os.path.join("mllm_evaluations", args.generated_imgs_dir.split('/')[-1] + "_results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline using MLLMs.")
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="datasets/test_datasets",
        required=False,
        help="Where the style and subject images are stored",
    )
    parser.add_argument(
        "--generated_imgs_dir",
        type=str,
        default="qualitative",
        required=False,
        help="Where the generated images are stored",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        required=False,
        help="The size of the images",
    )
    parser.add_argument(
        "--num_eval_images",
        type=int,
        default=1000,
        required=False,
        help="Number of generated images per scenario to use for MLLM evaluation",
    )
    parser.add_argument(
        "--dbg",
        type=int,
        default=0,
        required=False,
        help="If we are using interactive debugging mode",
    )

    args = parser.parse_args()

    if args.dbg == 1:
        model, tokenizer, image_processor, device = get_model(args)
    else:
        main(args)
