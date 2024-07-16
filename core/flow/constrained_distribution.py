
import torch as th
from core.flow.base_distribution import BaseDistribution
from core.constraints import BaseConstraint

class ConstrainedDistribution(BaseDistribution):
    """
    An un-normalized mollified probability distribution, based on cv signal. 
    When there are multiple constraint signals, `aggregate_method` describe how to combine them.
    """
    cv_aggregators = {
        'sum': lambda x: th.sum(th.clip(x, min=0), dim=1),
        'max': lambda x: th.max(th.clip(x, min=0), dim=1)[0]
    }

    def __init__(self, constraint: BaseConstraint, mollifier_sigma, aggregate_method: str="sum"):
        super().__init__()
        self.mollifier_sigma = mollifier_sigma
        self.constraint = constraint
        self.noise_prior = th.distributions.Normal(0, 1)
        self.aggregate_method = aggregate_method

    def log_prob(self, values: th.Tensor, y: th.Tensor) -> th.Tensor:
        cv_all  = self.constraint.get_cv(values, y)
        cv = self.cv_aggregators[self.aggregate_method](cv_all)
        return self.noise_prior.log_prob(cv/self.mollifier_sigma)