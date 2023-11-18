import seaborn
import pandas as pd
import numpy as np
from typing import List
import time
from experiments.problems import all_problems, BaseProblem
from core.constraints import CombinedConstraint
from core.sample_generation.hmc import get_hmc_samples_for_constraint
from experiments.common.setup_experiment import setup_experiment, flush_logs
import dataclasses



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
        if problem.state_action_bound_constraint is not None:
            constraint = CombinedConstraint(problem.constraint.var_count, problem.constraint.dim - problem.constraint.var_count, [problem.constraint, problem.state_action_bound_constraint])
        else:
            constraint = problem.constraint
        s = get_hmc_samples_for_constraint(constraint, params.count, 0) 
        print(f"Sample generation time for {k}: {(time.time() - start_time):.2f} seconds")
        if params.plot:
            df = pd.DataFrame({f"Dim {i}": s[:, i] for i in range(s.shape[1])})
            seaborn.pairplot(df, plot_kws={"s": 1}).savefig(f"{log_dir}/{k}.png")
        np.save(f"{log_dir}/{k}.npy", s)
        flush_logs()

if __name__ == "__main__":
    main()