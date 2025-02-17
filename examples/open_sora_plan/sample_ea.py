from videosys import OpenSoraPlanConfig, VideoSysEngine
prompts = [
    "a black dog wearing halloween costume", # animal
    "an apartment building with balcony", # archi
    "freshly baked finger looking cookies", # food
    "people carving a pumpkin", # human
    "scenic video of sunset", # scenery
]

sora_prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.",
    "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.",
    "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    "Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle. The art style is 3D and realistic, with a focus on lighting and texture. The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
    "A gorgeously rendered papercraft world of a coral reef, rife with colorful fish and sea creatures.",
]

def save_ref_video(video, i, prompt, engine):
    import os
    import torch
    # Check if `video` is already a PyTorch tensor
    if not isinstance(video, torch.Tensor):
        # Convert to a PyTorch tensor if it's not already one
        video = torch.tensor(video)
    ref_video_folder = f"/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/category_ref_videos_4s"
    # Save the video tensor to a .pt file

    ref_video_path = os.path.join(ref_video_folder, f"{i}.pt")
    torch.save(video, ref_video_path)

    print(f"Video saved as {ref_video_path}")

def run_base(save_ref_videos=False, load_ea_timesteps=False):
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v120", transformer_type="93x480p", num_gpus=1)
    engine = VideoSysEngine(config)

    ea_timesteps_list = []
    # prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." # "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    # File path
    prompt_file_path = "/home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt"

    # Read all prompts
    with open(prompt_file_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    for i, prompt in enumerate(prompts):

        if load_ea_timesteps:
            import yaml
            # Load YAML file
            ea_timesteps_path = "/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/outputs/93x480p_step40_search100_category/ea_timesteps.yaml"
            with open(ea_timesteps_path, "r") as file:
                ea = yaml.safe_load(file)  # Use safe_load to avoid execution risks

            ea_timesteps_list = ea["ea_timesteps_list"]
        
        if ea_timesteps_list:
            from pathlib import Path
            import os
            # Original YAML file path
            ea_path = Path(ea_timesteps_path)

            # Construct new folder path
            videos_folder = ea_path.parent / "videos_4s"

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
                
                video_filename = f"{i:04d}.mp4"  # Format index as 4 digits (e.g., 0000, 0001, etc.)
                video_save_path = os.path.join(videos_folder, video_filename)
                engine.save_video(video, video_save_path)
                print(f"Saved video with EA timesteps to {video_save_path}")
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
            engine.save_video(video, f"./outputs/category_4s_org_osp120/{prompt_suffix}_4s.mp4")




if __name__ == "__main__":
    run_base(save_ref_videos=False, load_ea_timesteps=True)

