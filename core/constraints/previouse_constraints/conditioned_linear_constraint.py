import torch as th
from dataclasses import dataclass
from typing import Callable, Tuple
from core.constraints.conditioned_constraint import ConditionedConstraint
from core.constraints.linear_constraint import LinearConstraint


@dataclass
class ConditionedLinearConstraint(ConditionedConstraint):
    get_a_b: Callable[[th.Tensor], Tuple[th.Tensor, th.Tensor]]

    def get_cv(self, x: th.Tensor) -> th.Tensor:
        # print(x.shape)
        assert x.shape[1] > self.conditional_param_count
        x, conditional_part = self.get_var_conditional_parts_separated(x)
        # print(">", x, conditional_part.shape, x.shape)
        A, b = self.get_a_b(conditional_part)
        # print("A>", A.shape, b.shape)
        return th.einsum('bij,bj->bi', A, x) - b

    
def test_conditioned_linear_constraint():
    cons = ConditionedLinearConstraint(2, 2, lambda c: (th.unsqueeze(c, dim=1), th.zeros(len(c), 1)))
    cv = cons.get_cv(th.tensor([[2, 3, 5 ,4], [11, 19, 43, 23]]))
    assert cv.shape == (2, 1)
    assert cv[0][0] ==  2*5 + 3*4
    assert cv[1][0] == 11*43 + 19*23