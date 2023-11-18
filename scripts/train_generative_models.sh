for p in "Reacher" "HalfCheetah" "Hopper" "Walker2d"; do
    echo "Running ${p}"
    python -m experiments.train_flow_forward --log_dir ./logs/generative_models/flow_forward/${p} --problem $p --data_file ./logs/sample_generation/1000000/${p}.npy --device cuda:0  --batch_size 32 &
    python -m experiments.train_wgan --log_dir ./logs/generative_models/wgan/${p} --problem $p --data_file ./logs/sample_generation/1000000/${p}.npy --device cuda:0  --batch_size 32 &
done 