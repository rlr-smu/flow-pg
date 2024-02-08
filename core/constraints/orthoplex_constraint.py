from typing import Tuple
from numpy import ndarray
import gurobipy as gp
from torch import Tensor
import torch as th
from core.constraints import BaseConstraint
from core.constraints.base_constraint import conditional_type

class OrthoplexConstraint(BaseConstraint):
    def __init__(self, var_count, ub: float):
        super().__init__(var_count, var_count)
        self.ub = ub

    def _get_cv(self, x: Tensor, y: conditional_type) -> Tensor:
        return th.abs(x*y).sum(dim=1).unsqueeze(dim=1) - self.ub
    
    def _add_gp_constraints(self, model, x: list, y: ndarray):
        abs_vars = []
        for i in range(self.a_dim):
            mul_var = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS)
            model.addConstr(mul_var == self.scale[i]*x[i]*y[i])
            abs_var = model.addVar(lb=0, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS)
            model.addGenConstrAbs(abs_var, mul_var)
            abs_vars.append(abs_var)
        model.addConstr(sum(abs_vars) <= self.max_power)