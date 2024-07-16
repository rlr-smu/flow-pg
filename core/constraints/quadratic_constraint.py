from dataclasses import dataclass
import torch as th
from core.constraints.base_constraint import BaseConstraint
import gurobipy as gp
from abc import ABC, abstractmethod
from typing import Tuple



class ConditionedQuadraticConstraint(BaseConstraint, ABC):
    def __init__(self, var_count, conditional_param_count):
        super().__init__(var_count, conditional_param_count)


    @abstractmethod
    def _get_q_m(y:th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        pass
    
    def _get_cv(self, x, y):
        Q, m = self._get_q_m(y)
        return th.einsum('bijk,bj,bk->bi', Q, x, x) - m

    def _add_gp_constraints(self, model, x, y):
        Q, m = self._get_q_m(th.tensor(y, device=self.device))
        Q, m = Q[0], m[0]
        for b in range(len(Q)):
            Sq = gp.QuadExpr()
            for i in range(len(Q[b])):
                for j in range(len(Q[b,i])):
                    Sq+=Q[b,i,j].item()*x[i]*x[j]
            model.addConstr(Sq <= m[b])
    

class QuadraticConstraint(BaseConstraint):
    """
    Quadratic constraints with: $\sum Q_ij a_i a_j \leq m$ 
    """

    def __init__(self, var_count, Q: th.Tensor, m: th.Tensor):
        super().__init__(var_count, 0)
        self.register_buffer('Q', Q)    
        self.register_buffer('m', m)
    
    def _get_cv(self, x, y):
        return th.einsum('ijk,bj,bk->bi', self.Q, x, x) - self.m

    def _add_gp_constraints(self, model, x, y):
        Q, m = self.Q, self.m
        for b in range(len(Q)):
            Sq = gp.QuadExpr()
            for i in range(len(Q[b])):
                for j in range(len(Q[b,i])):
                    Sq+=Q[b,i,j].item()*x[i]*x[j]
            model.addConstr(Sq <= m[b])


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