from dataclasses import dataclass
import torch as th
from copy import deepcopy
from typing import List, Union, Tuple
from core.constraints.base_constraint import BaseConstraint
from core.constraints.quadratic_constraint import QuadraticConstraint
from core.constraints.linear_constraint import LinearConstraint


IndexesType = List[Tuple[th.LongTensor, Union[th.LongTensor, None]]]
class CombinedConstraint(BaseConstraint):
    """Combine a sequence of constraints, as a union"""


    def __init__(self, var_count:int, conditional_param_count: int, constraints: List[BaseConstraint], indexes: IndexesType):
        super().__init__(var_count, conditional_param_count)
        print("c", constraints)
        self.register_module('constraints', th.nn.ModuleList(constraints))
        self.indexes = indexes
        if indexes is not None:
            assert len(indexes) == len(constraints), "Invalid indexes count"

    def _get_cv(self, x, y) -> th.Tensor:
        cvs = []
        for constraint, (xidx, yidx) in zip(self.constraints, self.indexes):
            x_part = x[:, xidx] 
            y_part = y[:, yidx] if yidx is not None else None # y can have a non values
            cvs.append(constraint.get_cv(x_part, y_part))
        return th.concat(cvs, dim=1)
    
    def _add_gp_constraints(self, model, x, y):
        for constraint, (xidx, yidx) in zip(self.constraints, self.indexes):
            x_part = x[xidx]
            y_part = y[yidx] if yidx is not None else None
            constraint.add_gp_constraints(model, x_part, y_part)

        
def test_combined_constraint():
    circle = QuadraticConstraint(2, th.eye(2).unsqueeze(0), th.tensor([1])) # circle with a unit radius
    l1 = LinearConstraint(1, -th.Tensor([[1]]), th.Tensor([0])) # v > 0
    constraints = [circle, l1, l1]
    indexes =  [(th.tensor([0, 1]), None), 
                (th.tensor([0]), None), 
                (th.tensor([1]), None)]
    combined_cons = CombinedConstraint(2, constraints, indexes) # circle and x0>0 and x1>0
    x = th.tensor([[0.1, 0.1], [-0.1, 0.1]])
    cv = combined_cons.get_cv(x)
    feasibility = combined_cons.is_feasible(x)
    assert feasibility.shape == (2, )
    assert feasibility[0] == True
    assert feasibility[1] == False
    




