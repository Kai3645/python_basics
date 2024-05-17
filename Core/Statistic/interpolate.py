import numpy as np
import scipy.interpolate

from Core.Basic import errInfo


def spline_1d(x, y, expand: int = 2, k: int = 3):
	"""
	:param x:
	:param y:
	:param expand:
	:param k:
		2 -> 'quadratic',
		3 -> 'cubic',
	:return: interFunc
	"""
	data_length = len(x)
	assert expand < data_length, errInfo(">> err, need more data for interpolation ..")

	length = data_length + expand * 2
	x_ex = np.zeros(length)
	x_ex[expand:length - expand] = x
	x_ex[:expand] = x[0] * 2 - x[expand:0:-1]
	x_ex[-expand:] = x[-1] * 2 - x[-2:-2 - expand:-1]

	y_ex = np.zeros(length)
	y_ex[expand:length - expand] = y
	y_ex[:expand] = y[expand:0:-1]
	y_ex[-expand:] = y[-2:-2 - expand:-1]

	return scipy.interpolate.interp1d(x_ex, y_ex, k)
