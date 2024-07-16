from dataclasses import dataclass
from experiments.common.setup_experiment import flush_logs, get_value_logger, get_log_dir
from core.constraints import BaseConstraint, BoxConstraint
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedDistribution
from hydra.utils import instantiate
import torch as th
from experiments.config import TaskConfig
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import hydra
import optuna
import os
from omegaconf import DictConfig, OmegaConf
import time
import logging

@dataclass
class Cfg:
    task: TaskConfig
    data_file: str
    train_sample_count: int 
    test_sample_count: int
    epochs: int 
    eval_freq: int 
    device: str 
    lr: float 
    batch_size: int 
    hidden_size: int 
    transform_count: int 
    feasibility_error_margin: float
    mollifier_sigma: float 
    gradient_clip_value: float 
    take_log_again: bool 
    save_sample_data: bool
    plot: bool

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="train_flow_forward")
def main(cfg: Cfg):
    """
    Train flow with forward KL divergence using generated samples from a file.
    """
    
    print(OmegaConf.to_yaml(cfg))
    logger.info("datafile: "+ cfg.data_file)
    log_dir = get_log_dir()
    value_logger = get_value_logger()

    # Get the constraint
    constraint:BaseConstraint  = instantiate(cfg.task.action_constraint, _convert_="all")
    constraint = constraint.to(cfg.device)
    conditional_p_count = constraint.conditional_param_count

    # Define the flow model
    flow = RealNvp(constraint.var_count, cfg.transform_count, conditional_param_count=conditional_p_count, hidden_size=cfg.hidden_size,).to(cfg.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=cfg.lr)

    # Define the mollified uniform distribution
    box_l = th.full((constraint.var_count, ), -1).float()
    box_h = th.full((constraint.var_count, ), 1).float()
    uniform_constraint = BoxConstraint(constraint.var_count, box_l, box_h).to(cfg.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, cfg.mollifier_sigma)

    # Load dataset
    data = th.from_numpy(np.load(cfg.data_file)).float().to(cfg.device)
    if cfg.train_sample_count + cfg.test_sample_count > len(data):
        raise ValueError("Not enough samples in the dataset")
    train_data = data[:cfg.train_sample_count]
    test_data = data[cfg.train_sample_count: cfg.test_sample_count+ cfg.train_sample_count]

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    value_logger.log("Staring experiment")

    os.makedirs(log_dir + "/figures", exist_ok=True)
    # sinkhorn_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    np.save(f"{log_dir}/figures/test_data.npy", test_data.cpu().numpy())
    value_logger.log(f"Test data: {len(test_data)}")
    start_time = time.time()
    f1 = 0
    try:
        for epoch in range(cfg.epochs):
            losses = []
            # Update flow for each batch
            for batch, in data_loader:
                x, y = batch[:, :constraint.var_count], batch[:, constraint.var_count:]
                loss = update_flow_batch(flow, mollified_uniform_distribution, x, y, optimizer, gradient_clip_value=cfg.gradient_clip_value, take_log_again=cfg.take_log_again)
                losses.append(loss)

            value_logger.log(f"Updated batch: {epoch}")
            if (epoch+1)%cfg.eval_freq == 0:
                # Evaluate
                with th.no_grad():
                    # Calculate accuracy (z -(g)-> x)
                    z = th.rand((len(test_data), constraint.var_count)).float().to(cfg.device)*2-1
                    x, y = test_data[:, :constraint.var_count], test_data[:, constraint.var_count:]
                    # z = th.concat([z_act, test_data[:, constraint.var_count:]], dim=1) # Inclue conditional variables
                    generated_samples = flow.g(z, y)[0]
                    validity = constraint.is_feasible(generated_samples, y, cfg.feasibility_error_margin)
                    valid_count = validity.int().sum().item()
                    accuracy = valid_count/len(validity)
                    generated_samples_actions = generated_samples[:, :constraint.var_count]
                    value_logger.log(f"{generated_samples_actions.shape}, {test_data.shape}")

                    # Calculate recall (x -(f)-> z)
                    mapped_z = flow.f(x, y)[0][:, :constraint.var_count]
                    validity_z = th.all(mapped_z >= -1, dim=1) & th.all(mapped_z <= 1, dim=1)
                    valid_z_count = validity_z.int().sum().item()
                    recall = valid_z_count/len(validity_z)
                
                    if cfg.save_sample_data:
                        np.save(f"{log_dir}/figures/{epoch+1}.npy", generated_samples.cpu().numpy())
                    if cfg.plot:
                        fig = constraint.plot(generated_samples_actions)
                        fig.savefig(f"{log_dir}/figures/{epoch+1}.png") 

                if accuracy + recall > 0:
                    f1 = accuracy*recall*2/(accuracy+recall)
                else:
                    f1 = 0

                elapsed_time = time.time() - start_time
                value_logger.record("train/time_elapsed", elapsed_time)
                value_logger.record("train/mean_loss", np.mean(losses))
                value_logger.record("train/accuracy", accuracy)
                value_logger.record("train/recall", recall)
                value_logger.record("train/f1", f1)

                value_logger.log(f"Epoch: {epoch+1}: Mean loss {np.mean(losses):.4f}, Acc: {accuracy*100: .2f}%, Recall: {recall*100: .2f}%")
                flow.save_module(f"{log_dir}/model.pt")
                value_logger.record("train/epoch", epoch+1)
                flush_logs()
                value_logger.dump(epoch)
    except Exception:
        logger.exception("message")
    return f1


if __name__ == "__main__":
    main()