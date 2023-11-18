from dataclasses import dataclass
import torch as th
from core.constraints.linear_constraint import LinearConstraint, BaseConstraint

@dataclass
class BoxConstraint(BaseConstraint):
    """
    Bound constraint with lower and higer bound for each dimension"""
    low: th.Tensor
    high: th.Tensor

    def __post_init__(self):
        assert self.dim == len(self.low) == len(self.high)
        l_A = -th.eye(self.dim, dtype=self.high.dtype)
        l_b = -self.low
        h_A = th.eye(self.dim, dtype=self.low.dtype)
        h_b = self.high
        A = th.concat([l_A, h_A])
        b = th.concat([l_b, h_b])
        self.liniear_constraint = LinearConstraint(self.var_count, A, b)
    
    def get_cv(self, x: th.Tensor) -> th.Tensor:
        cv = self.liniear_constraint.get_cv(x)
        return cv


def test_box_constraint():
    cons = BoxConstraint(2, low=th.tensor([2, 3]).double(), high=th.tensor([10, 15]).double())
    x = th.tensor([[5, 5], [0, 0], [12, 4]]).double()
    cv = cons.get_cv(x)
    assert cv.shape == (3, 4)
    assert cv[0][0] == 2-5
    assert cv[0][1] == 3-5
    assert cv[0][2] == 5-10
    assert cv[0][3] == 5-15
    feasibility = cons.is_feasible(x)
    assert feasibility.shape == (3,)
    assert feasibility[0] == True
    assert feasibility[1] == False
    assert feasibility[2] == False