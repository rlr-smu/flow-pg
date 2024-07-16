from typing import Union
import torch as th
from numpy import ndarray
from collections import defaultdict
from stable_baselines3.common.logger import Logger
from core.constraints import BaseConstraint
import logging
import numpy as np

log = logging.getLogger(__name__)

class ConstrainedAlgorithm:
    logger: Logger
    state_indexes: Union[list, None]
    constraint: BaseConstraint

    def __init__(self, constraint:BaseConstraint, state_indexes: Union[list, None], cv_error_margin:float, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.constraint = constraint
        self.cv_error_margin = cv_error_margin
        self.state_indexes = state_indexes
        self.cv_metrices = defaultdict(lambda :0)

    def project_if_infeasible(self, action: ndarray, observation: ndarray, record_key: str = None) -> ndarray:
        if self.state_indexes is not None:
            y = self.constraint.to_tensor(observation[:, self.state_indexes]) 
        else: 
            y = None
        x = self.constraint.to_tensor(action)
        cv = th.sum(th.clip(self.constraint.get_cv(x, y), min=0), dim=1)
        feasibility = cv <= self.cv_error_margin

        if record_key is not None:
            self.cv_metrices[f"cv/{record_key}_cv_count"] += (~feasibility).sum().item()
            self.cv_metrices[f"cv/{record_key}_cv_value"] += cv.sum().item()
            for k in self.cv_metrices:
                self.logger.record(k, self.cv_metrices[k])

        for i, f in enumerate(feasibility):
            if not f:
                y = observation[i, self.state_indexes] if self.state_indexes is not None else None
                action[i] = self.constraint.project(action[i], y).clip(min=-1, max=1)
        return np.array(action)