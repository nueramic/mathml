import unittest
from src.nueramic_mathml.calculus import *

test_functions = {
    'test_1': {
        'x0': torch.tensor([1., 2.]),
        'func': lambda x: (x ** 2).sum(),
        'answer': torch.tensor([2., 4.])
    }
}


class CalculusFunctionsTest(unittest.TestCase):
    def test_gradient(self):
        for test in test_functions:
            self.assertTrue(torch.allclose(gradient()))


if __name__ == '__main__':
    unittest.main()
