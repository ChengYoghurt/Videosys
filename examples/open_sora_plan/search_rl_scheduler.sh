CUDA_VISIBLE_DEVICES=0 \
python examples/open_sora_plan/search_ea.py \
--outdir 'examples/open_sora_plan/outputs/step50_search100' \
--n_samples 6 \
--num_sample 1000 \
--time_step 50 \
--max_epochs 10 \
--population_num 5 \
--mutation_num 2 \
--crossover_num 1 \
--seed 1024 \
--use_ddim_init_x false \
--ref_videos '/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/ref_videos' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \