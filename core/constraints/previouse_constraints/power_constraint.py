from dataclasses import dataclass
import torch as th
# from core.constraints.conditioned_constraint import ConditionedConstraint

@dataclass
class PowerConstraint(ConditionedConstraint):
    """
    State-dependent Action Constraints with the from
    $`\sum max{w_i a_i, 0} \leq M, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """
    ub: float
    def get_cv(self, x: th.Tensor) -> th.Tensor:
        return th.clip(x[:, :self.var_count]*x[:, self.var_count:], min=0).sum(dim=1).unsqueeze(dim=1) - self.ub

@dataclass
class OrthoplexConstraint(ConditionedConstraint):
    """
    State-dependent Action Constraints with the from
    $`\sum |w_i a_i| \leq M, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """
    ub: float
    def get_cv(self, x: th.Tensor) -> th.Tensor:
        return th.abs(x[:, :self.var_count]*x[:, self.var_count:]).sum(dim=1).unsqueeze(dim=1) - self.ub

@dataclass
class OrthoplexConstraintLB(ConditionedConstraint):
    """
    State-dependent Action Constraints with the from
    $`\sum |w_i a_i| \geq lb, |a_i| \leq 1`$ where $w_i$ is a velocity corresponding to $a_i$
    """
    lb: float
    def get_cv(self, x: th.Tensor) -> th.Tensor:
        return self.lb - th.abs(x[:, :self.var_count]*x[:, self.var_count:]).sum(dim=1).unsqueeze(dim=1) 