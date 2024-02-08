from dataclasses import dataclass
import torch as th
from core.constraints.base_constraint import BaseConstraint

@dataclass
class ConditionedConstraint(BaseConstraint):
    """Conditional constraint conditioned on the latter part of the input and produce 
    """
    conditional_param_count: int

    def get_var_conditional_parts_separated(self, x: th.Tensor) -> th.Tensor:
        return x[:, :self.var_count], x[:, self.var_count:]
    
    @property
    def dim(self):
        return self.var_count + self.conditional_param_count