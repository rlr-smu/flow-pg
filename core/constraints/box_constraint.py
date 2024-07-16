from dataclasses import dataclass
import torch as th
from core.constraints.linear_constraint import LinearConstraint, BaseConstraint


class BoxConstraint(LinearConstraint):
    """
    Bound constraint with lower and higer bound for each dimension"""
    def __init__(self, var_count: int, low: float, high: float):
        if isinstance(high, (int, float)):
            low, high = th.tensor(low).repeat(var_count), th.tensor(high).repeat(var_count)

        assert var_count == len(low) == len(high)
        l_A = -th.eye(var_count, dtype=high.dtype)
        h_A = th.eye(var_count, dtype=low.dtype)
        A = th.concat([l_A, h_A])
        b = th.concat([-low, high])
        super().__init__(var_count, A, b)


def test_box_constraint():
    cons = BoxConstraint(2, low=3., high=10.)
    x = th.tensor([[5, 5], [0, 0], [12, 4]]).float()
    cv = cons.get_cv(x, None)
    assert cv.shape == (3, 4)
    assert cv[0][0] == 3-5
    assert cv[0][1] == 3-5
    assert cv[0][2] == 5-10
    assert cv[0][3] == 5-10
    feasibility = cons.is_feasible(x, None, 0.)
    assert feasibility.shape == (3,)
    assert feasibility[0] == True
    assert feasibility[1] == False
    assert feasibility[2] == False

    assert BoxConstraint(2, low=-30., high=30.).is_feasible(th.tensor([[0.,0.]]), None, 0)[0]
    assert BoxConstraint(6, low=-30., high=30.).is_feasible(th.tensor([[ 0.2606,  0.2082, -0.1028, -0.0217, -0.3268,  0.1907]]), None, 0)[0]