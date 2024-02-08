import torch as th 
from core.constraints.quadratic_constraint import QuadraticConstraint

class SphereConstraint(QuadraticConstraint):
    def __init__(self, var_count, ub: float):
        Q = th.eye(var_count, dtype=th.float).unsqueeze(0)
        m = th.tensor([ub], dtype=th.float)
        super().__init__(var_count, Q, m)