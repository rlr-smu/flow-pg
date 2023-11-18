import torch as th
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseConstraint(ABC):
    """Base class for all constraints.
    """
    var_count: int

    @abstractmethod
    def get_cv(self, x: th.Tensor) -> th.Tensor:
        """
        Return constraint violation signal for each constraint, positive: violated, negative: not-violated"""
        pass

    @property
    def dim(self):
        return self.var_count
    
    def is_feasible(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.dim
        return th.sum(th.clip(self.get_cv(x), min=0), dim=1) <= 0

    def to(self, device: str):
        """Not an effecient implementation, don't call allways. Do it at the begining of the setup."""
        self = deepcopy(self)
        for k in dir(self):
            element = getattr(self, k)
            if hasattr(element, 'to'):
                try:
                    setattr(self, k, element.to(device))
                except TypeError:
                    pass
        return self