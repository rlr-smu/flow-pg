from numpy import ndarray
from core.constraints.base_constraint import BaseConstraint
from dataclasses import dataclass
from typing import Union
import torch as th

class LinearConstraint(BaseConstraint):

    """
    Linear constraints with:
        Ax <= b,
    """
    def __init__(self, var_count: int, A: th.Tensor, b: th.Tensor):
        super().__init__(var_count, 0)
        assert A.shape[0] == len(b)
        self.register_buffer('A', A)
        self.register_buffer('b', b)

    def _get_cv(self, x: th.TensorType, y):
        return th.einsum('ij,bj->bi', self.A, x) - self.b
    
    
    def _add_gp_constraints(self, model, x: list, y: ndarray):
        for i in range(len(self.A)):
            model.addConstr(th.einsum('j,j', self.A[i], x) <= self.b[i])

def test_linear_constraints():
    cons = LinearConstraint(2, th.tensor([[1, 1]]), th.tensor([10]))
    cv = cons.get_cv(th.tensor([[2, 3], [10, 10]]), None)
    assert cv.shape == (2, 1)
    assert cv[0][0] == -5
    assert cv[1][0] == 10

    cons = LinearConstraint(2, th.tensor([[15, 3],[2,3]]), th.tensor([10, 2]))
    cv = cons.get_cv(th.tensor([[4, 3], [1, 2]]), None)
    assert cv.shape == (2, 2)
    assert cv[0][0] == (15*4+3*3 - 10)
    assert cv[0][1] == (2*4+3*3 - 2)
    assert cv[1][0] == (1*15+2*3 - 10)
    assert cv[1][1] == (1*2+2*3 - 2)