from dataclasses import dataclass
import torch as th
from core.constraints.base_constraint import BaseConstraint

@dataclass
class QuadraticConstraint(BaseConstraint):
    """
    Quadratic constraints with: $\sum Q_ij a_i a_j \leq m$ 
    """
    Q: th.TensorType
    m: th.TensorType

    def get_cv(self, x: th.TensorType):
        assert x.shape[1] == self.Q.shape[1], "Shape should match with Q"
        return th.einsum('ijk,bj,bk->bi', self.Q, x, x) - self.m

    def __post_init__(self):
        self.Q.shape[0] == len(self.m), "Q and b should have same number of rows"
        assert self.Q.shape[1] == self.Q.shape[2], "Q should be square"

def test_quadratic_constraints():
    cons = QuadraticConstraint(2, th.tensor([
        [[1, 0],
        [0,1]],
        [[1, 2],
        [100,0]]
        ]), th.tensor([10, 5]))
    cv = cons.get_cv(th.tensor([[4, 3], [1, 2]]))
    assert cv.shape == (2, 2)
    assert cv[0][0] == (4*4+3*3 - 10)
    assert cv[0][1] == (4*4+2*4*3 + 100*4*3 - 5)
    assert cv[1][0] == (1*1+2*2 - 10)
    assert cv[1][1] == (1*1+ 2*1*2 + 100*1*2 - 5)