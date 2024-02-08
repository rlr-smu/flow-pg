import numpy as np
from pyhmc import hmc
from typing import Union
import matplotlib.pyplot as plt
import logging
from core.constraints.base_constraint import BaseConstraint
from core.constraints.quadratic_constraint import QuadraticConstraint
import torch as th



def get_hmc_samples(logprob, v_count, sample_count, random_seed: Union[None, int]=None, origin: np.ndarray=None):
    if random_seed:
        np.random.seed(random_seed)
    
    x0 = np.zeros(v_count)
    if origin is not None:
        x0 = origin
    x0 =  x0 + (np.random.rand(v_count)-0.5)* 0.000001
    return hmc(logprob, x0=x0, n_samples=sample_count)

def get_hmc_samples_for_constraint(constraint: BaseConstraint, state_bound_constraint: BaseConstraint, sample_count, random_seed: Union[None, int]=None, origin: np.ndarray=None):
    constraint = constraint.to(th.float64)
    state_bound_constraint = state_bound_constraint.to(th.float64) if state_bound_constraint is not None else None

    def logprob(x):
        grad = np.zeros((len(x), ))
        if constraint.conditional_param_count > 0:
            y = th.tensor(x[constraint.var_count:]).unsqueeze(dim=0)
            feasible = state_bound_constraint.is_feasible(y, None, 0)[0]
        else:
            feasible = True
            y = None
        feasible = feasible and constraint.is_feasible(th.tensor(x[: constraint.var_count]).unsqueeze(dim=0), y, 0)[0]
        if not feasible:
            return -np.inf, grad
        else:
            return 0.0, grad
    samples = get_hmc_samples(logprob, constraint.var_count + constraint.conditional_param_count, sample_count, random_seed, origin)
    if constraint.conditional_param_count > 0:
        y = th.tensor(samples[:, constraint.var_count:])
    else:
        y = None
    feasibility = constraint.is_feasible(th.tensor(samples[:, :constraint.var_count]), y, 0)
    feasible_count = feasibility.int().sum()
    if feasible_count < sample_count:
        print(f"Warning: Fewer feasible samples were generated with HMC: {feasible_count}/{sample_count}")
    return samples[feasibility.numpy()]

    
def test_hmc_samples():
    Q = th.eye(2).unsqueeze(dim=0).double()
    m =  th.tensor([1]).double()
    cons = QuadraticConstraint(2, Q, m)
    sampels = get_hmc_samples_for_constraint(cons, 1000)
    assert len(sampels) == 1000
    s = sampels[:, 0]**2 + sampels[:, 1]**2 <= 1
    assert np.all(s)