from videosys import OpenSoraPlanConfig, VideoSysEngine
prompts = [
    "a black dog wearing halloween costume", # animal
    "an apartment building with balcony", # archi
    "freshly baked finger looking cookies", # food
    "people carving a pumpkin", # human
    "scenic video of sunset", # scenery
]

def run_base(save_ref_videos=False, load_ea_timesteps=False):
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v120", transformer_type="29x480p", num_gpus=1)
    engine = VideoSysEngine(config)

    ea_timesteps_list = []
    # prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." # "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    for i, prompt in enumerate(prompts):
        if load_ea_timesteps:
            import yaml
            # Load YAML file
            ea_timesteps_path = "/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/outputs/step50_search100/ea_timesteps.yaml"
            with open(ea_timesteps_path, "r") as file:
                ea = yaml.safe_load(file)  # Use safe_load to avoid execution risks

            ea_timesteps_list = ea["ea_timesteps_list"]
        
        if ea_timesteps_list:
            from pathlib import Path
            import os
            # Original YAML file path
            ea_path = Path(ea_timesteps_path)

            # Construct new folder path
            videos_folder = ea_path.parent / "videos"

            # Create the folder if it doesn't exist
            videos_folder.mkdir(parents=True, exist_ok=True)
            for idx, ea_timesteps in enumerate(ea_timesteps_list):
                video = engine.generate(
                    prompt=prompt,
                    guidance_scale=7.5,
                    num_inference_steps=100,
                    seed=1024,
                    ea_timesteps=ea_timesteps
                ).video[0]
                
                prompt_suffix = prompt[:20] if len(prompt) > 20 else prompt
                video_save_path = os.path.join(videos_folder, f"{prompt_suffix}_{idx}.mp4")
                engine.save_video(video, video_save_path)
                print(f"Saved video with EA timesteps to {os.path.join(videos_folder, f'{prompt_suffix}_{i}.mp4')}")
        else:
            print("Generating videos WITHOUT EA...")
            video = engine.generate(
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=100,
                seed=1024,
            ).video[0]

            if save_ref_videos:
                print(f"Saving reference video {i} for prompt '{prompt}'")
                save_ref_video(video, i, prompt, engine)

            prompt_suffix = prompt[:20] if len(prompt) > 20 else prompt
            engine.save_video(video, f"./outputs/{prompt_suffix}.mp4")

def save_ref_video(video, i, prompt, engine):
    import os
    import torch
    # Check if `video` is already a PyTorch tensor
    if not isinstance(video, torch.Tensor):
        # Convert to a PyTorch tensor if it's not already one
        video = torch.tensor(video)
    ref_video_folder = f"/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/ref_videos"
    # Save the video tensor to a .pt file

    ref_video_path = os.path.join(ref_video_folder, f"{i}.pt")
    torch.save(video, ref_video_path)

    print(f"Video saved as {ref_video_path}")




if __name__ == "__main__":
    run_base(save_ref_videos=False, load_ea_timesteps=True)

