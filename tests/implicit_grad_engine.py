import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest, dataclasses
from implicit_gradients.implicit_grad_engine import *
from utils import *

class TestIterativeConjugateGradientEngine(BasicObjectTest):
    CLASS  = IterativeConjugateGradientEngine
    KWARGS = dict(proximal_regularization_strength = 1)

class TestNeumannInverseHessianApproximationEngine(BasicObjectTest):
    CLASS  = NeumannInverseHessianApproximationEngine
    KWARGS = dict(num_inverse_hvp_iters = 10)

def main(): unittest.main()
if __name__ == '__main__': main()
