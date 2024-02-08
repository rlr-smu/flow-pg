import argparse
from dataclasses import dataclass
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
from experiments.problems import all_problems, BaseProblem
from core.constraints import BaseConstraint, BoxConstraint
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedDistribution
import torch as th
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time


def main():
    """
    Train flow forwad using generated samples from a file.
    """
    
    @dataclass
    class Options:
        problem: str
        data_file: str
        train_sample_count: int = 500_000
        test_sample_count: int = 500_000
        epochs: int = 500
        eval_freq: int = 1
        device: str = 'cpu'
        lr: float = 1e-5
        batch_size: int = 256
        hidden_size: int = 256
        transform_count: int = 6
        mollifier_sigma: float = 0.0001
        gradient_clip_value: float = 0.1
        take_log_again: bool = False

    args = setup_experiment("train_flow_forward", Options)
    logger = get_value_logger(args.log_dir)
    params: Options = args.params

    # Get the constraint
    problem: BaseProblem = all_problems[params.problem]
    constraint:BaseConstraint  = problem.constraint
    constraint = constraint.to(params.device)
    conditional_p_count = getattr(constraint, 'conditional_param_count', 0)

    # Define the flow model
    flow = RealNvp(constraint.var_count, params.transform_count, conditional_param_count=conditional_p_count, hidden_size=params.hidden_size,).to(params.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=params.lr)

    # Define the mollified uniform distribution
    box_l = th.full((constraint.dim, ), -1).float()
    box_h = th.full((constraint.dim, ), 1).float()
    uniform_constraint = BoxConstraint(constraint.dim, box_l, box_h).to(params.device)
    mollified_uniform_distribution = ConstrainedDistribution(uniform_constraint, params.mollifier_sigma)

    # Load dataset
    data = th.from_numpy(np.load(params.data_file)).float().to(params.device)
    if params.train_sample_count + params.test_sample_count > len(data):
        raise ValueError("Not enough samples in the dataset")
    train_data = data[:params.train_sample_count]
    test_data = data[params.train_sample_count: params.test_sample_count+ params.train_sample_count]

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    print("Staring experiment")

    os.makedirs(args.log_dir + "/figures", exist_ok=True)
    # sinkhorn_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    np.save(f"{args.log_dir}/figures/test_data.npy", test_data.cpu().numpy())
    print("Test data", len(test_data))
    start_time = time.time()
    for epoch in range(params.epochs):
        losses = []
        # Update flow for each batch
        for batch, in data_loader:
            loss = update_flow_batch(flow, mollified_uniform_distribution, batch, optimizer, gradient_clip_value=params.gradient_clip_value, take_log_again=params.take_log_again)
            losses.append(loss)

        print(f"Updated batch: {epoch}")
        if (epoch+1)%params.eval_freq == 0:
            # Evaluate
            with th.no_grad():
                # Calculate accuracy (z -(g)-> x)
                z_act = th.rand((len(test_data), constraint.var_count)).float().to(params.device)*2-1
                z = th.concat([z_act, test_data[:, constraint.var_count:]], dim=1) # Inclue conditional variables
                generated_samples = flow.g(z)[0]
                validity = constraint.is_feasible(generated_samples)
                valid_count = validity.int().sum().item()
                accuracy = valid_count/len(validity)
                generated_samples_actions = generated_samples[:, :constraint.var_count]
                print(generated_samples_actions.shape, test_data.shape)
                # sinkhorn_loss = sinkhorn_loss_func(generated_samples_actions, test_data[:, :constraint.var_count]).mean().item()

                # Calculate recall (x -(f)-> z)
                mapped_z = flow.f(test_data)[0][:, :constraint.var_count]
                validity_z = th.all(mapped_z >= -1, dim=1) & th.all(mapped_z <= 1, dim=1)
                valid_z_count = validity_z.int().sum().item()
                recall = valid_z_count/len(validity_z)
            
                # fig = problem.plot(generated_samples)
                # fig.savefig(f"{args.log_dir}/figures/{epoch+1}.png")
                # np.save(f"{args.log_dir}/figures/{epoch+1}.npy", generated_samples.cpu().numpy())

            elapsed_time = time.time() - start_time
            logger.record("train/time_elapsed", elapsed_time)
            logger.record("train/mean_loss", np.mean(losses))
            logger.record("train/accuracy", accuracy)
            logger.record("train/recall", recall)



            print(f"Epoch: {epoch+1}: Mean loss {np.mean(losses):.4f}, Acc: {accuracy*100: .2f}%, Recall: {recall*100: .2f}%")
            flow.save_module(f"{args.log_dir}/model.pt")
            logger.record("train/epoch", epoch+1)
            flush_logs()
            logger.dump(epoch)


if __name__ == "__main__":
    main()