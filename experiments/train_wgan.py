import argparse
from dataclasses import dataclass
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
from experiments.problems import all_problems, BaseProblem
from core.constraints import BaseConstraint, BoxConstraint
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedDistribution
from core.common.loadable_module import LoadbleModule
import torch as th
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np
import os

class Generator(nn.Module, LoadbleModule):
    def __init__(self, latent_dim, conditional_param_count):
        self.kwargs = {"conditional_param_count": conditional_param_count, "latent_dim": latent_dim}
        self.latent_dim = latent_dim
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+conditional_param_count, 32, normalize=False),
            *block(32, 64),
            nn.Linear(64, latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


    
class Discriminator(nn.Module):
    def __init__(self, latent_dim, conditional_param_count):
        super(Discriminator, self).__init__()
        self.kwargs = {"conditional_param_count": conditional_param_count, "latent_dim": latent_dim}
        self.model = nn.Sequential(
            nn.Linear(latent_dim + conditional_param_count, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, sample):
        return self.model(sample)



def main():
    """
    Train wgan using generated samples from a file.
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
        n_critic: int = 10
        sample_interval: float = 10.
        gradient_clip_value: float = 10

    args = setup_experiment("train_wgan", Options)
    params: Options = args.params

    # Get the constraint
    problem:BaseProblem  = all_problems[params.problem]
    constraint = problem.constraint
    constraint = constraint.to(params.device)
    conditional_p_count = getattr(constraint, 'conditional_param_count', 0)
    latent_dim = constraint.var_count
    logger = get_value_logger(args.log_dir)
    
 


    generator = Generator(latent_dim, conditional_p_count).to(device=params.device)
    discriminator = Discriminator(latent_dim, conditional_p_count).to(device=params.device)

    # Optimizers
    optimizer_G = th.optim.RMSprop(generator.parameters(), lr=params.lr)
    optimizer_D = th.optim.RMSprop(discriminator.parameters(), lr=params.lr)

    
    # Load dataset
    data = th.from_numpy(np.load(params.data_file)).to(params.device).to(th.float32)
    if params.train_sample_count > len(data):
        raise ValueError("Not enough samples in the dataset")
    train_data = data[:params.train_sample_count]
    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    test_data = data[params.train_sample_count:params.train_sample_count + params.test_sample_count]
    os.makedirs(args.log_dir + "/figures", exist_ok=True)
    np.save(f"{args.log_dir}/figures/test_data.npy", test_data.cpu().numpy())
    print("Test data", len(test_data))
    batches_done = 0
    for epoch in range(params.epochs):
        losses_G = []
        losses_D = []
        # Update flow for each batch
        for i, (batch,) in enumerate(data_loader):

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(th.Tensor(np.random.uniform(-1, 1, (batch.shape[0], latent_dim), ),).to(params.device))

            # Generate a batch of images
            conditional_vars = batch[:, constraint.var_count:] # Get conditional variables for generator
            z = th.concat([z, conditional_vars], dim=1)
            fake_samples = generator(z).detach()
            fake_samples = th.concat([fake_samples, conditional_vars], dim=1)
            # Adversarial loss
            loss_D = -th.mean(discriminator(batch)) + th.mean(discriminator(fake_samples))

            loss_D.backward()
            optimizer_D.step()
            losses_D.append(loss_D.item())

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-params.gradient_clip_value, params.gradient_clip_value)

            # Train the generator every n_critic iterations
            if i % params.n_critic == 0:
                optimizer_G.zero_grad()
                # Generate a batch of samples
                gen_samples = generator(z)
                gen_samples = th.concat([gen_samples, conditional_vars], dim=1)
                # Adversarial loss
                loss_G = -th.mean(discriminator(gen_samples))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, params.epochs, batches_done % len(data_loader), len(data_loader), loss_D.item(), loss_G.item())
                )            


                losses_G.append(loss_G.item())
            batches_done += 1

        if (epoch+1)%params.eval_freq == 0:
            # Evaluate
            with th.no_grad():
                # Calculate accuracy (z -(g)-> x)
                z = Variable(th.Tensor(np.random.uniform(-1, 1, (params.test_sample_count, latent_dim)))).to(params.device)
                conditional_vars = test_data[:, constraint.var_count:]
                z = th.concat([z, conditional_vars], dim=1)
                generated_samples_actions = generator(z)
                generated_samples = th.concat([generated_samples_actions, conditional_vars], dim=1)
                validity = constraint.is_feasible(generated_samples.double())
                valid_count = validity.int().sum().item()
                accuracy = valid_count/len(validity)
                fig = problem.plot(generated_samples)
                fig.savefig(f"{args.log_dir}/figures/{epoch+1}.png")
                np.save(f"{args.log_dir}/figures/{epoch+1}.npy", generated_samples.cpu().numpy())

                # sinkhorn_loss = sinkhorn_loss_func(generated_samples_actions, test_data[:, :problem.var_count]).mean().item()
                # Cannot calculate recall

            # print(f"Epoch: {epoch+1}: Descriminator mean loss: {np.mean(losses_D):.4f}, Generator mean loss: {np.mean(losses_G):.4f}, Acc: {accuracy*100: .2f}%, "")
            logger.record("train/mean_d_loss", np.mean(losses_D))
            logger.record("train/mean_g_loss", np.mean(losses_G))
            logger.record("train/accuracy", accuracy)
            logger.record("train/epoch", epoch+1)
            logger.dump(epoch)
            generator.save_module(f"{args.log_dir}/generator.pt")
            flush_logs()



if __name__ == "__main__":
    main()