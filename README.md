# FlowPG
Source Code for "FlowPG: Action-constrained Policy Gradient with Normalizing Flows"
```
@inproceedings{brahmanage2023flowpg,
  title={FlowPG: Action-constrained Policy Gradient with Normalizing Flows},
  author={Brahmanage, Janaka Chathuranga and Ling, Jiajing and Kumar, Akshat},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Generating samples
### HMC
```
python -m experiments.generate_samples_with_hmc --problems All --count 1000000 --log_dir logs/sample_generation/hmc
```
Results wtill be saved in `logs/samples_generation/hmc*`. 

### Rejection sampling
```bash
python -m experiments.generate_samples_with_rejection --problems All --count 1000000 --log_dir logs/sample_generation/rejection
```
Results wtill be saved in `logs/samples_generation/rejection*`. 

### PySDD
To generate data for BSS-3 and BSS-5. Results are saved in `./logs/sample_generation/sdd`
```bash
python ./experiments/generate_sdd.py 
```

## Training the flow model

### Flow Model
```bash
python -m experiments.train_flow_forward --log_dir ./logs/generative_models/flow_forward/Reacher --problem Reacher --data_file ./logs/sample_generation/1000000/Reacher.npy --device cuda:0  --batch_size 32
```

### WGAN
```bash
python -m experiments.train_wgan --log_dir ./logs/generative_models/wgan/Reacher --problem Reacher --data_file ./logs/sample_generation/1000000/Reacher.npy --device cuda:0  --batch_size 32
```


## Training the RL agent
Coming soon :rocket: ..
