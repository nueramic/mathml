import torch

from src.nueramic_mathml import *

test = {
    'function': lambda x: (3 * x[0] ** 2 + 0.5 * x[1] ** 2),
    'x0': torch.arange(0, 3).double(),
    'keep_history': True,
    'max_iter': 10,
}

test_functions_ineq_constr = [
    (
        lambda x: - torch.cos(x).sum(),
        torch.tensor([-0.4, 1], dtype=torch.float64),
        [lambda x: x[0] + 1, lambda x: x[1] + 2]
    ),
    (
        lambda x: x.sum(),
        torch.tensor([1., 1.], dtype=torch.float64),
        [lambda x: x[0], lambda x: x[1]]
    ),
    (
        lambda x: (x[0] + 1) ** 2 + x[1] ** 2,
        torch.tensor([0.3, 0.1], dtype=torch.float64),
        [lambda x: 10 - x.abs().sum()],
    ),
    (
        lambda x: (x[0] + 1) ** 2 + x[1] ** 2,
        torch.tensor([0.2, 0.7], dtype=torch.float64),
        [lambda x: 1 - x.abs().sum()]
    )
]

if __name__ == '__main__':
    print(bfgs(**test)[1])
    print(gd_constant(**test)[1])
    print(gd_frac(**test)[1])
    print(gd_optimal(**test)[1])
    print(primal_dual_interior(*test_functions_ineq_constr[3], keep_history=True))
    print(log_barrier_solver(*test_functions_ineq_constr[3], keep_history=True))
