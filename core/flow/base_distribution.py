from abc import ABC, abstractmethod
import torch as th

class BaseDistribution(ABC):

    @abstractmethod
    def log_prob(self, values: th.Tensor) -> th.Tensor:
        pass