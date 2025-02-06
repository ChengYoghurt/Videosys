CUDA_VISIBLE_DEVICES=0 \
python examples/open_sora_plan/search_ea.py \
--outdir 'examples/open_sora_plan/outputs/step50_search100' \
--n_samples 6 \
--num_sample 1000 \
--time_step 50 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 1024 \
--use_ddim_init_x false \
--ref_latent '/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/ref_videos/generated_video.pt' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \