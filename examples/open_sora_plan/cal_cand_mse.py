from videosys import OpenSoraPlanConfig, VideoSysEngine
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 4s
# prompts = [
#     "a muffin with a burning candle and a love sign by a ceramic mug", # food
#     "a group of friend place doing hand gestures of agreement", # humannvi
#     "aerial view of snow piles", # scenery
#     "yacht sailing through the ocean", # vehicle
# ]

# 1s
prompts = [
    "a black dog wearing halloween costume", # animal
    "an apartment building with balcony", # archi
    "freshly baked finger looking cookies", # food
    "people carving a pumpkin", # human
]

def load_ref_videos(ref_videos_folder):
    ref_videos = []
    for i in range(4):
        video_path = os.path.join(ref_videos_folder, f"{i}.pt")
        video = torch.load(video_path)
        video_normalized = video.float() / 255.0
        ref_videos.append(video_normalized)
    return ref_videos

def get_cand_mse(cand, engine, ref_videos, output_dir, device='cuda'):
    mse_scores = []
    for i, prompt in enumerate(prompts):
        cand_video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=100,
        seed=1024,
        ea_timesteps=cand,
        ).video[0]
        cand_video_float = cand_video.float() / 255.0
        ref_video_float = ref_videos[i] # normalized alrd
        mse_loss = F.mse_loss(cand_video_float, ref_video_float)
        print(f"MSE={mse_loss} for prompt={prompt}")
        mse_scores.append(mse_loss.item())

        # == save the cand video ==
        video_filename = f"{prompt}.mp4"
        videos_folder = output_dir / "mse_cand_videos"
        videos_folder.mkdir(parents=True, exist_ok=True)
        video_save_path = os.path.join(videos_folder, video_filename)
        engine.save_video(video, video_save_path)
        print(f"Saved video with EA timesteps to {video_save_path}")
    
    mean_mse = np.mean(mse_scores)
    print("Mean MSE Loss:", mean_mse)
    return mean_mse

def main():
    config = OpenSoraPlanConfig(version="v120", transformer_type="29x480p", num_gpus=1)
    engine = VideoSysEngine(config)

    # == load ref videos ==
    ref_vidoes_dir = "examples/open_sora_plan/assets/ref_videos_1s"
    ref_videos = load_ref_videos(ref_vidoes_dir)

    # == load ea timesteps ==
    import yaml
    # Load YAML file
    ea_timesteps_path = "examples/open_sora_plan/outputs/29x480p_step50_search100_cache/ea_timesteps.yaml"
    ea_path = Path(ea_timesteps_path)
    videos_folder = ea_path.parent
    with open(ea_timesteps_path, "r") as file:
        ea = yaml.safe_load(file)  # Use safe_load to avoid execution risks
        ea_timesteps_list = ea["ea_timesteps_list"]

        for idx, cand in enumerate(ea_timesteps_list):
            mean_mse = get_cand_mse(cand=cand, engine=engine, ref_videos=ref_videos, output_dir=videos_folder)
            print(f"NO.{idx} EA MSE={mean_mse}")

if __name__ == "__main__":
    main()