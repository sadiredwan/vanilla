import numpy as np
from typing import Callable
from numpy import ndarray

def deriv(func: Callable[[ndarray], ndarray],
	input_: ndarray,
	delta: float = 0.001) -> ndarray:
	'''
	Evaluates the derivative of a function "func" at every element in the
	"input_" array.
	'''
	return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)