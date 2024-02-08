import torch as th
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union 
import gurobipy as gp


conditional_type = Union[th.Tensor, None]
conditional_type_np = Union[np.ndarray, None]

class BaseConstraint(th.nn.Module, ABC):
    """Base class for all constraints, including conditional constraints.
    Notation: 
        x - point in the target space
        z - point in the latent space
        y - point in the conditional space
    """

    def __init__(self, var_count:int, conditional_param_count:int):
        super().__init__()
        self.var_count = var_count # Number of variables in the constraint. i.e. = len(x), len(z)
        self.conditional_param_count = conditional_param_count # Number of variables in the conditional part. i.e. = len(y)

    @abstractmethod
    def _get_cv(self, x: th.Tensor, y:conditional_type) -> th.Tensor:
        """Implement by all concrete classes"""
        pass

    @abstractmethod
    def _add_gp_constraints(self, model, x:list, y:np.ndarray):
        """Implemented for individual instance, not for tha batch"""
        pass

    def get_cv(self, x: th.Tensor, y:conditional_type) -> th.Tensor:
        """
        Return constraint violation signal for each constraint, positive: violated, negative: not-violated"""
        self.__validate_input(x, y)
        return self._get_cv(x, y)
    
    def __validate_input(self, x_or_z: Union[th.Tensor, np.ndarray], y: Union[th.Tensor, np.ndarray]):
        instance_dim = 1
        assert x_or_z.shape[instance_dim] == self.var_count, "x shape should match with var_count"
        assert (y is None and self.conditional_param_count==0) or (y is not None and y.shape[instance_dim] == self.conditional_param_count), \
                                    f"y shape should match with conditional_param_count, y:{y}, conditional_param_count:{self.conditional_param_count}"
 
    def is_feasible(self, x: th.Tensor, y: conditional_type, cv_error_margin: float) -> th.Tensor:
        return th.sum(th.clip(self.get_cv(x, y), min=0), dim=1) <= cv_error_margin

    def add_gp_constraints(self, model: gp.Model, x:list, y: conditional_type_np):
        """Needs to implement for each constraint."""
        assert len(x) == self.var_count
        assert (y is None and self.conditional_param_count==0) or y.shape[0] == self.conditional_param_count
        return self._add_gp_constraints(model, x, y)
    
    def project(self, x_input: np.ndarray, y: conditional_type_np) -> np.ndarray:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()
            with gp.Model(env=env) as model:
                # model.params.NonConvex = 2
                x = []
                for _ in range(self.var_count):
                    x.append(model.addVar(lb=-1, ub =1, vtype = gp.GRB.CONTINUOUS))
                obj = gp.QuadExpr()
                for i in range(self.var_count):
                    obj+=(x[i]-x_input[i])**2
                model.setObjective(obj, sense = gp.GRB.MINIMIZE)
                self.add_gp_constraints(model, x, y)
                model.optimize()
                x_value = np.array(model.X[0:self.var_count])
                return x_value