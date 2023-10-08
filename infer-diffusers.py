import argparse
import time
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import os

#from diffusers import AutoencoderTiny

from torchmetrics.functional.multimodal import clip_score
from functools import partial


def main(args):
    print(args)
    torch.set_num_threads(args.num_cores)
    distilled = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_safetensors=True,
        #"nota-ai/bk-sdm-small", use_safetensors=True,
        #"openvino-sd-v1-4-distil", use_safetensors=True,
    )


    #distilled.vae = AutoencoderTiny.from_pretrained(
    #    "sayakpaul/taesd-diffusers", use_safetensors=True,
    #)

    distilled.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)

    seed = 0
    #generator = torch.manual_seed(seed)

    NUM_ITERS_TO_RUN = 1
    NUM_INFERENCE_STEPS = args.num_inference_steps
    NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt

    prompts = ["a golden vase with different flowers"]
    start = time.time_ns()
    for _ in range(NUM_ITERS_TO_RUN):
        images = distilled(
            prompts,
            num_inference_steps=NUM_INFERENCE_STEPS,
            #generator=generator,
            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
            output_type="numpy",
        ).images
    end = time.time_ns()


    distilled_sd = f"{(end - start) / 1e6:.1f}"
    print(f"Execution time -- {distilled_sd} ms\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")
    # randomizer params
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # scheduler params
    parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")
    # diffusion params
    parser.add_argument("--num-cores", type=int, default=20, help="num cores")
    parser.add_argument("--num-inference-steps", type=int, default=25, help="num inference steps")
    parser.add_argument("--num-images-per-prompt", type=int, default=1, help="num images per prompt")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")
    # prompt
    parser.add_argument("--prompt", type=str, default="Street-art painting of Emilia Clarke in style of Banksy, photorealism", help="prompt")
    # Parameter re-use:
    parser.add_argument("--params-from", type=str, required=False, help="Extract parameters from a previously generated image.")
    # img2img params
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str, default="output.png", help="output image name")
    args = parser.parse_args()
    main(args)

#clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
#def calculate_clip_score(images, prompts):
#    images_int = (images * 255).astype("uint8")
#    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
#    return round(float(clip_score), 4)
#
#sd_clip_score = calculate_clip_score(images, prompts * NUM_IMAGES_PER_PROMPT)
#print(f"CLIP score: {sd_clip_score}")


