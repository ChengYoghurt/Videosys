PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python examples/open_sora_plan/search_ea.py \
--outdir 'examples/open_sora_plan/outputs/step50_search100_category4' \
--n_samples 6 \
--num_sample 1000 \
--time_step 50 \
--max_epochs 10 \
--population_num 10 \
--mutation_num 5 \
--crossover_num 2 \
--seed 1024 \
--use_ddim_init_x false \
--ref_videos '/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/ref_videos_4s' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \