import torch

from src.nueramic_mathml import gd_constant_step, gd_frac_step, gd_optimal_step, bfgs

test = {
    'function': lambda x: (3 * x[0] ** 2 + 0.5 * x[1] ** 2),
    'x0': torch.arange(0, 3).double(),
    'keep_history': True,
    'max_iter': 10,
}

if __name__ == '__main__':
    print(bfgs(**test)[1])
    print(gd_constant_step(**test)[1])
    print(gd_frac_step(**test)[1])
    print(gd_optimal_step(**test)[1])
