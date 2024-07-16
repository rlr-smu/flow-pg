import torch as th 
from core.constraints.quadratic_constraint import QuadraticConstraint
import numpy as np
from matplotlib import pyplot as plt

class SphereConstraint(QuadraticConstraint):
    def __init__(self, var_count, ub: float):
        Q = th.eye(var_count, dtype=th.float).unsqueeze(0)
        m = th.tensor([ub], dtype=th.float)
        super().__init__(var_count, Q, m)


def test_spehere_constraint():
    c = SphereConstraint(2, 0.05)
    original = np.array([[-0.5, -0.1], [0.5, 0.4], [0.9, 0.9], [-0.9, 0.9]])
    feasiblity= c.is_feasible(c.to_tensor(original), None, 0)
    projected = [c.project(original[i], None) for i in range(original.shape[0])]
    print(projected, feasiblity)
    fig, ax = plt.subplots() 
    ax.scatter(original[:, 0], original[:, 1], label='Original') 
    ax.scatter([p[0] for p in projected], [p[1] for p in projected], label='Projected') 
    ax.set_xlabel('X') 
    ax.set_ylabel('Y') 
    ax.legend()
    fig.savefig('./outputs/sphere_constraint.png')
