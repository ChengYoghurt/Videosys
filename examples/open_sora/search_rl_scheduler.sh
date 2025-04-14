PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=9 \
python examples/open_sora/search_ea.py \
--outdir 'examples/open_sora/outputs/4sx480p_step21_search30_cache' \
--n_samples 6 \
--num_sample 1000 \
--time_step 21 \
--max_epochs 50 \
--population_num 50 \
--mutation_num 20 \
--crossover_num 15 \
--seed 1024 \
--use_ddim_init_x false \
--ref_videos 'examples/open_sora/assets/4sx480p' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \