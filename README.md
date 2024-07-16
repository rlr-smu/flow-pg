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
Hyper param tuning
```bash
python -m experiments.train_flow_forward task=reacher +data_file=outputs/generated_data/reacher.npy
```


## Training the RL agent
DDPG+Projection
```
python -m experiments.train_rl task=reacher agent=ddpg_pro
```

**FlowPG**: Place the trained flow model inside `models/{env_name}.pt`. Eg. `models/reacher.pt` and run the following command.

```
python -m experiments.train_rl task=reacher agent=flowpg
```