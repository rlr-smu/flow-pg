import seaborn
import pandas as pd
import numpy as np
from typing import List
import time
from experiments.problems import all_problems, BaseProblem, BaseConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core.constraints import CombinedConstraint
from experiments.common.setup_experiment import setup_experiment, flush_logs
import dataclasses
import torch as th


def get_samples_with_rejection(problem: BaseProblem, sample_count: int):
    s_c = 0
    batch_size = 100000
    valid_samples = []
    while s_c <= sample_count:
        data_loader = problem.get_state_data_loader(batch_size, 1, 'cpu', 0)
        for state in data_loader:
            actions = th.rand((batch_size, problem.constraint.var_count))*2 - 1
            values = th.concat([actions,state], dim=1)
            validity = problem.constraint.is_feasible(values)
            valid = values[validity]
            valid_samples.append(valid)
            s_c += len(valid)
    return th.concat(valid_samples, dim=0)
    

def plot_3d(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    dimNames = [f"Dim {i}" for i in range(3)]
    x, y, z = [df[n] for n in dimNames]


    ax.set_xlabel(dimNames[0])
    ax.set_ylabel(dimNames[1])
    ax.set_zlabel(dimNames[2])
    ax.scatter(x, y, z, s=0.1, alpha=0.1) 
    return fig


def main():
    @dataclasses.dataclass
    class Options:
        problems: List[str]
        count: int = 1000
        plot: bool = False

    args = setup_experiment("sample_generation_hmc", Options) 

    log_dir = args.log_dir
    params = args.params
    problems = all_problems.keys() if "All" in params.problems else params.problems
    for k in problems:
        if k not in all_problems:
            raise ValueError(f"Invalid problem name: {k}")

    print("Running data generation for:", ", ".join(problems))
    for k in problems:
        print("Running:", k)
        problem:BaseProblem  = all_problems[k]
        start_time = time.time()

        s = get_samples_with_rejection(problem, params.count) 
        print(f"Sample generation time for {k}: {(time.time() - start_time):.2f} seconds")
        df = pd.DataFrame({f"Dim {i}": s[:, i] for i in range(s.shape[1])})
        if params.plot:
            if s.shape[1] == 3:
                fig = plot_3d(df)
            else:
                fig = seaborn.pairplot(df, plot_kws={"s": 1})
            fig.savefig(f"{log_dir}/{k}.png")
        np.save(f"{log_dir}/{k}.npy", s)
        flush_logs()

if __name__ == "__main__":
    main()