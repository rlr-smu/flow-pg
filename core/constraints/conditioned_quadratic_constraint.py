import torch as th
from dataclasses import dataclass
from typing import Callable, Tuple
from core.constraints.conditioned_constraint import ConditionedConstraint


@dataclass
class ConditionedQuadraticConstraint(ConditionedConstraint):
    get_q_m: Callable[[th.Tensor], Tuple[th.Tensor, th.Tensor]]

    def get_cv(self, x: th.Tensor) -> th.Tensor:
        x, conditional_part = self.get_var_conditional_parts_separated(x)
        Q, m = self.get_q_m(conditional_part)
        assert (Q.shape[2] == Q.shape[3] == x.shape[1]), "Q should be square and match x dim"
        assert Q.shape == (len(x), m.shape[1], x.shape[1], x.shape[1])
        return th.einsum('bijk,bj,bk->bi', Q, x, x) - m

    
def test_conditioned_quadratic_constraint():
    def get_qm(c):
        """Here we stack the conditional part twice to get the Q square matric"""
        return (th.stack([c, c], dim=1).unsqueeze(dim=1), th.ones(len(c), 1))

    cons = ConditionedQuadraticConstraint(2, 2, get_qm)
    cv = cons.get_cv(th.tensor([[2, 3, 5 ,4], [11, 19, 43, 23]]))
    assert cv.shape == (2, 1)
    assert cv[0][0] ==  2*2*5 + 2*3*(4+5) + 3*3*4 - 1
    assert cv[1][0] ==  11*11*43 + 11*19*(43+23) + 19*19*23 - 1