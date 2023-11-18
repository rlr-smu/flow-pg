from core.constraints import QuadraticConstraint, BoxConstraint, ConditionedQuadraticConstraint, CombinedConstraint, OrthoplexConstraint, ConditionedLinearConstraint, PowerConstraint, BaseConstraint, OrthoplexConstraintLB
import torch as th
from typing import Union, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.flow.data_loaders import UniformDataLoader, GaussianDataLoader, CombinedDataLoader


def get_box_constraint(dim):
    return BoxConstraint(dim, th.tensor([-1]*dim).double(), th.tensor([1]*dim).double())



class BaseProblem:
    constraint: BaseConstraint
    state_action_bound_constraint: Union[BaseConstraint, None] = None # Used to sample using HMC 
    action_plot_range: list = [-1, 1]
    state_dist: Literal["Uniform", "Gaussian"] = "Uniform"
    state_dist_u_bound: float = 1 # For uniform: i.e. [-1, 1]
    state_dist_g_mu: float = 0.
    state_dist_g_sigma: float = 1. # For gaussian

    @property
    def action_bound_constraint(self):
        return get_box_constraint(self.constraint.var_count) # var_count=action_count
    
    def plot(self, samples: th.Tensor):
        samples = samples.cpu().numpy()
        if samples.shape[1] == 2 or samples.shape[1] > 4:
            # print(samples.shape, )
            figure, ax = plt.figure(figsize=(5, 5)), plt.gca()
            H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=(80, 80), range=np.array([self.action_plot_range, self.action_plot_range]))
            plot = ax.pcolormesh(xedges, yedges, H, cmap=plt.cm.jet, )
            ax.get_figure().colorbar(plot)
            return figure
        else:
            import seaborn
            df = pd.DataFrame({f"Dim {i}": samples[:, i] for i in range(samples.shape[1])})
            return seaborn.pairplot(df, plot_kws={"s": 1})
    
    def get_state_data_loader(self, batch_size, batch_count, device, seed):
        conditional_parm_count = getattr(self.constraint, 'conditional_param_count', 0)
        if self.state_dist == "Uniform":
            return UniformDataLoader(batch_size, batch_count, conditional_parm_count, device, seed, (-self.state_dist_u_bound, self.state_dist_u_bound))
        if self.state_dist == "Gaussian":
            return GaussianDataLoader(batch_size, batch_count, conditional_parm_count, device, seed, self.state_dist_g_mu, self.state_dist_g_sigma)
        else:
            raise ValueError("Invalid state-dist")
    
"""
Define all the constraints used in all experiments here."""

class R_L2(BaseProblem):
    constraint = QuadraticConstraint(2, th.eye(2).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    action_plot_range: list = [-0.27, 0.27]
    

class HC_O(BaseProblem):
    high_b = th.tensor([1]*6 + [30]*6).double()
    state_action_bound_constraint = BoxConstraint(12, low=-high_b, high=high_b)
    constraint = OrthoplexConstraint(6, 6, 20)
    state_dist = "Gaussian"
    state_dist_g_sigma = 15

def get_power_constraint(dim:int, ub, state_bound):
    high_b = th.tensor([1]*dim + [state_bound]*dim).double()
    bounds = BoxConstraint(2*dim, low=-high_b, high=high_b)
    base = PowerConstraint(dim, dim, ub)
    return base, bounds


class H_M(BaseProblem):
    state_dist_u_bound = 10
    constraint, state_action_bound_constraint = get_power_constraint(3, 10, state_dist_u_bound)


class W_M(BaseProblem):
    state_dist_u_bound = 10
    constraint, state_action_bound_constraint = get_power_constraint(6, 10, state_dist_u_bound)


all_problems = {
    "Reacher": R_L2(),
    "HalfCheetah": HC_O(),
    "Hopper": H_M(),
    "Walker2d": W_M(),
}
