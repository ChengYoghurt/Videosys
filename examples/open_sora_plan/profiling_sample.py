from videosys import OpenSoraPlanConfig, VideoSysEngine
import torch
import csv

# List of different video modes (frame length and resolution)
video_modes = [
    {"frame_length": 29, "resolution": "480p"},
    {"frame_length": 29, "resolution": "720p"},
    {"frame_length": 93, "resolution": "480p"},
    {"frame_length": 93, "resolution": "720p"},
]

# Open a CSV file to log timings
csv_filename = "timing_results.csv"
fieldnames = ["generation_num", "frame_length", "resolution", "tot_other_time", "tot_self_attn_time", "tot_cross_attn_time", "tot_ff_time", "inference_time"]

def run_base(mode, gen_num):
    # Open the CSV file in append mode to write the timings
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Create a header if the file is empty
        if file.tell() == 0:  # If the file is empty, write the header
            writer.writeheader()

        try:
            # Configuration for OpenSoraPlan
            config = OpenSoraPlanConfig(
                version="v120", 
                transformer_type=f"{mode['frame_length']}x{mode['resolution']}", 
                num_gpus=1
            )
            engine = VideoSysEngine(config)

            prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage."

            # Timing events
            inf_start = torch.cuda.Event(enable_timing=True)
            inf_end = torch.cuda.Event(enable_timing=True)

            # Record start time for inference
            inf_start.record()

            # Run the generate method and get the timing results
            generate_result = engine.generate(
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=100,
                seed=-1,
            )

            video = generate_result[0].video[0]
            timings = generate_result[1]

            # Record end time for inference
            inf_end.record()
            torch.cuda.synchronize()

            # Get the total inference time
            inference_time = inf_start.elapsed_time(inf_end)

            # Create timings_dict with the results from this run
            timings_dict = {
                "tot_other_time": timings["tot_other_time"],
                "tot_self_attn_time": timings["tot_self_attn_time"],
                "tot_cross_attn_time": timings["tot_cross_attn_time"],
                "tot_ff_time": timings["tot_ff_time"],
                "generation_num": gen_num + 1,  # Track the generation number (1-based index)
                "frame_length": mode["frame_length"],
                "resolution": mode["resolution"],
                "inference_time": inference_time  # Add the total inference time
            }

            # Print the results for debugging/confirmation
            print(f"Mode: {mode['frame_length']}x{mode['resolution']} | Generation: {gen_num + 1}")
            print(f"Inference Time: {inference_time} ms")
            print(f"Timing Results: {timings_dict}")

            # Save the video output
            engine.save_video(video, f"./outputs/{prompt[:10]}_{mode['frame_length']}x{mode['resolution']}_gen{gen_num+1}.mp4")

            # Write the timings_dict into the CSV file
            writer.writerow(timings_dict)
        except Exception as e:
            print(f"Error processing mode {mode['frame_length']}x{mode['resolution']} generation {gen_num + 1}: {str(e)}")

if __name__ == "__main__":
    # Iterate through the video modes
    for mode in video_modes:
        for gen_num in range(10):  # Call run_base 10 times for each mode
            run_base(mode, gen_num)
