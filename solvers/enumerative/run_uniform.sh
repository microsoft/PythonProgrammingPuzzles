python solve_challenges.py \
    -p "../../problems/*.json" \
    -m uniform \
    --solve_uniform \
    --copy_prob 0.5 \
    --max_n_progs 10000 \
    --timeout_secs 3600 \
    --threads 40 \
    --logging_dir results/uniform \
    --out_file results/uniform/out.json 
