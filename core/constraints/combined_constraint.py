from dataclasses import dataclass
import torch as th
from copy import deepcopy
from typing import List, Union
from core.constraints.base_constraint import BaseConstraint
from core.constraints.conditioned_constraint import ConditionedConstraint
from core.constraints.quadratic_constraint import QuadraticConstraint
from core.constraints.linear_constraint import LinearConstraint

@dataclass
class CombinedConstraint(ConditionedConstraint):
    """Combine a sequence of constraints, as a union"""
    constraints: List[BaseConstraint]
    indexes: Union[List[th.LongTensor], None] = None

    def get_cv(self, x: th.Tensor) -> th.Tensor:
        if self.indexes is None:
            cvs = [self.constraints[i].get_cv(x) for i in range(len(self.constraints))]
        else:
            cvs = [self.constraints[i].get_cv(x[:, self.indexes[i]]) for i in range(len(self.constraints))]
        return th.concat(cvs, dim=1)

    def to(self, device: str):
        self = deepcopy(self)
        self.constraints = [v.to(device) for v in self.constraints]
        if self.indexes is not None:
            self.indexes = [v.to(device) for v in self.indexes]
        return super().to(device)

        
def test_combined_constraint():
    circle = QuadraticConstraint(2, th.eye(2).unsqueeze(0), th.tensor([1])) # circle with a unit radius
    l1 = LinearConstraint(1, -th.Tensor([[1]]), th.Tensor([0])) # v > 0
    constraints = [circle, l1, l1]
    indexes =  [th.tensor([0, 1]), th.tensor([0]), th.tensor([1])]
    combined_cons = CombinedConstraint(2, constraints, indexes) # circle and x0>0 and x1>0

    x = th.tensor([[0.1, 0.1], [-0.1, 0.1]])
    cv = combined_cons.get_cv(x)
    feasibility = combined_cons.is_feasible(x)
    assert feasibility.shape == (2, )
    assert feasibility[0] == True
    assert feasibility[1] == False
    




