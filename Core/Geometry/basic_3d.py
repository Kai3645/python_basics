import math

import numpy as np

from Core.Basic import is_zero
from Core.Geometry.SpaceLine import SpaceLine
from Core.Geometry.SpacePlane import SpacePlane
from Core.Math import gaussian_elimination


def is_CCW(pts, nv):
	"""
	if is counterclockwise
	:param pts: points
	:param nv: normal vector
	:return:
	"""
	v1 = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]]
	v2 = [pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2]]
	return np.dot(np.cross(v1, v2), nv) >= 0


def is_CW(pts, nv):
	"""
	if is clockwise
	:param pts: points
	:param nv: normal vector
	:return:
	"""
	return not is_CCW(pts, nv)


def intersect_l2(line1: SpaceLine, line2: SpaceLine):
	"""
	X - V1 * t1 = P1
	X - V2 * t2 = P2
	| 1       -V10   0  | | x0 |   | P10 |
	|   ..      :    :  | | :  |   |  :  |
	|      1  -V1i   0  | | xi | = | P1i |
	| 1         0  -V20 | | t1 |   | P20 |
	|   ..      :   :   | | t2 |   |  :  |
	|      1    0  -V2i |          | P2i |
	:param line1:
	:param line2:
	:return:
	"""
	dim = 3
	M = np.zeros((dim * 2, dim + 3))
	idxes = np.arange(dim)
	M[idxes, idxes] = 1
	M[idxes + dim, idxes] = 1
	M[:dim, dim] = -line1.v
	M[dim:, dim + 1] = -line2.v
	M[:dim, -1] = line1.p
	M[dim:, -1] = line2.p
	x, dof, _ = gaussian_elimination(M)
	if x is None: return None
	return x[:dim]


def intersect_lp(line: SpaceLine, plane: SpacePlane):
	"""
	X - V1 * t1 = P1
	N2 * X = d2
	|  1           -V10 | | x0 |   | P10 |
	|      ..        :  | | :  | = |  :  |
	|           1  -V1i | | xi |   | P1i |
	| N20  ..  N2i   0  | | t1 |   | d2  |
	:param line:
	:param plane:
	:return:
	"""
	dim = 3
	M = np.zeros((dim + 1, dim + 2))
	idxes = np.arange(dim)
	M[idxes, idxes] = 1
	M[:dim, dim] = -line.v
	M[dim, :dim] = plane.nv
	M[:dim, -1] = line.v
	M[dim, -1] = plane.d
	x, dof, _ = gaussian_elimination(M)
	if x is None: return None
	return x[:dim]


def intersect_p2(plane1: SpacePlane, plane2: SpacePlane):
	"""
	N1 * X = d1
	N2 * X = d2
	| N10  ..  N1i | X = | d1 |
	| N20  ..  N2i |     | d2 |
	:param plane1:
	:param plane2:
	:return:
	"""
	dim = 3
	M = np.zeros((2, dim + 1))
	M[0, :dim] = plane1.nv
	M[1, :dim] = plane2.nv
	M[0, -1] = plane1.d
	M[1, -1] = plane2.d
	x, dof, W = gaussian_elimination(M, fast_mode = False)
	if x is None: return None
	return SpaceLine.init_pv(x, W[0])


def intersect_p3(plane1: SpacePlane, plane2: SpacePlane, plane3: SpacePlane):
	"""
	N1 * X = d1
	N2 * X = d2
	N3 * X = d3
	| N10  ..  N1i | | x0 |   | d1 |
	| N20  ..  N2i | | :  | = | d2 |
	| N30  ..  N3i | | xi |   | d2 |
	:param plane1:
	:param plane2:
	:param plane3:
	:return:
	"""
	dim = 3
	M = np.zeros((3, dim + 1))
	M[0, :dim] = plane1.nv
	M[1, :dim] = plane2.nv
	M[2, :dim] = plane3.nv
	M[0, -1] = plane1.d
	M[1, -1] = plane2.d
	M[3, -1] = plane3.d
	x, dof, _ = gaussian_elimination(M)
	return x


def line_division(pt1, pt2, density):
	"""
	:param pt1: dim array
 	:param pt2: dim array
	:param density: points per meters
	:return:
	"""
	v = [pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]]
	rr = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
	if is_zero(rr): return np.array(pt1, np.float64)
	total = math.ceil(math.sqrt(rr) * density)
	k = np.tile(np.linspace(0, 1, total), (3, 1)).transpose()
	return k * v + pt1


if __name__ == '__main__':
	def main():
		pt1 = [1, 1, 1]
		pt2 = [4, 5, 6]
		print(line_division(pt1, pt2, 10))
		pass


	main()
	pass
