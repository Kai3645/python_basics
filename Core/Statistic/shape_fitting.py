import math

import numpy as np

from Core.Basic import NUM_ERROR


def fitting_circle_2d(pts):
	"""
	( 2x  2y  1 ) ( a  b  r^2 - a^2 - b^2 )^T = ( x^2 + y^2 )
	c = r^2 - a^2 - b^2
	( a  b  c ) | 4Σxx  4Σxy  2Σx | = ( 2Σx(x^2 + y^2)  2Σy(x^2 + y^2)   Σ(x^2 + y^2) )
	            | 4Σxy  4Σyy  2Σy |
	            | 2Σx   2Σy    n  |
	X A = B
	X = B * A^-1
	:param pts:
	:return:
		x: center x
		y: center y
		r: radius
	"""
	length = len(pts)
	assert length > 2
	xx = pts[:, 0] * pts[:, 0]
	yy = pts[:, 1] * pts[:, 1]
	xy = pts[:, 0] * pts[:, 1]
	A = np.zeros((3, 3))
	A[0, 0] = np.sum(xx) * 4
	A[0, 1] = np.sum(xy) * 4
	A[0, 2] = np.sum(pts[:, 0]) * 2
	A[1, 0] = A[0, 1]
	A[1, 1] = np.sum(yy) * 4
	A[1, 2] = np.sum(pts[:, 1]) * 2
	A[2, 0] = A[0, 2]
	A[2, 1] = A[1, 2]
	A[2, 2] = length
	xx_yy = xx + yy
	B = np.zeros(3)
	B[0] = np.sum(xx_yy * pts[:, 0]) * 2
	B[1] = np.sum(xx_yy * pts[:, 1]) * 2
	B[2] = np.sum(xx_yy)
	if np.linalg.det(A) < NUM_ERROR: return 0, 0, 0
	a, b, c = B.dot(np.linalg.inv(A))
	r = math.sqrt(c + a * a + b * b)
	return a, b, r


def fitting_normal_2d(x, y):
	"""
	             N            -(x - μ)^2
	f(x) = ------------- exp(------------)
	        sqrt(2πσ^2)          2σ^2
	mean = μ
	sigma = σ
	K = N / sqrt(2πσ^2)
	:param x:
	:param y:
	:return:
	"""
	valid = y > np.max(y) / 4
	x = x[valid]
	y = y[valid]

	A = np.ones((len(x), 3))
	A[:, 1] = np.log(y)
	A[:, 2] = x * -2
	tA = A.transpose()
	inv_AA = np.linalg.inv(tA.dot(A))
	B = x * x * -1
	a, b, mean = inv_AA.dot(tA).dot(B)
	print(a, b, mean)
	sigma = math.sqrt(b / 2)  # todo: ValueError: math domain error
	K = math.exp((mean * mean - a) / b)
	return mean, sigma, K
