from core.constraints.base_constraint import BaseConstraint
from dataclasses import dataclass
from typing import Union
import torch as th

@dataclass
class LinearConstraint(BaseConstraint):
    """
    Linear constraints with:
        Ax <= b,
    """
    A: th.TensorType
    b: th.TensorType

    def get_cv(self, x: th.TensorType):
        return th.einsum('ij,bj->bi', self.A, x) - self.b

    def __post_init__(self):
        self.A.shape[0] == len(self.b)

def test_linear_constraints():
    cons = LinearConstraint(2, th.tensor([[1, 1]]), th.tensor([10]))
    cv = cons.get_cv(th.tensor([[2, 3], [10, 10]]))
    assert cv.shape == (2, 1)
    assert cv[0][0] == -5
    assert cv[1][0] == 10

    cons = LinearConstraint(2, th.tensor([[15, 3],[2,3]]), th.tensor([10, 2]))
    cv = cons.get_cv(th.tensor([[4, 3], [1, 2]]))
    assert cv.shape == (2, 2)
    assert cv[0][0] == (15*4+3*3 - 10)
    assert cv[0][1] == (2*4+3*3 - 2)
    assert cv[1][0] == (1*15+2*3 - 10)
    assert cv[1][1] == (1*2+2*3 - 2)