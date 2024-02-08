import seaborn
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from core.constraints import BoxConstraint, BaseConstraint
from core.sample_generation.hmc import get_hmc_samples_for_constraint
from experiments.common.setup_experiment import get_log_dir
import hydra
from hydra.utils import instantiate
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="generate_samples_with_hmc")
def main(cfg: DictConfig):
    log_dir = get_log_dir()
    cfg_const = cfg.task.constraint
    constraint:BaseConstraint  = instantiate(cfg_const.action_constraint, _convert_="all")

    # If state bounds are defined, we need to create a bound constraint for the state
    if cfg_const.state_bounds is not None:
        state_bound_constraint = BoxConstraint(constraint.var_count, cfg_const.state_bounds[0], cfg_const.state_bounds[1])
    else:
        state_bound_constraint = None
    count = cfg.count
    s = get_hmc_samples_for_constraint(constraint, state_bound_constraint, count, 0) 
    if cfg.plot:
        df = pd.DataFrame({f"Dim {i}": s[:, i] for i in range(s.shape[1])})
        seaborn.pairplot(df, plot_kws={"s": 1}).savefig(f"{log_dir}/plot.png")
    np.save(f"{log_dir}/data.npy", s)
    logger.info(f"Done, count:{len(s)}")


if __name__ == "__main__":
    main()