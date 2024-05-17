import math

import numpy as np
from scipy.special import fresnel

from Core.Basic import NUM_ERROR, is_zero


def in_triangle(points, vertex3):
	"""
	:param points: (n, 2) array like
	:param vertex3: (3, 2) array like
	:return: valid if in triangle
	"""
	wx = [vertex3[1][0] - vertex3[0][0], vertex3[1][1] - vertex3[0][1]]
	wy = [vertex3[2][0] - vertex3[0][0], vertex3[2][1] - vertex3[0][1]]
	wz = np.cross(wx, wy)
	assert not is_zero(wz), "err, invalid triangle .."
	P = np.atleast_2d(points)
	N, dim = P.shape
	assert dim == 2, f"err, dim{dim} over 2 .."
	P = P - vertex3[0]
	U = np.dot(P, np.linalg.inv([wx, wy]))
	valid = np.logical_and(U[:, 0] > -NUM_ERROR, U[:, 1] > -NUM_ERROR)
	valid = np.logical_and(valid, U[:, 0] + U[:, 1] < 1 + NUM_ERROR)
	if N > 1: return valid
	return valid[0]


def in_rectangle(points, vertex4):
	"""
	:param points: nx2 number
	:param vertex4: 4x2 number (loop like)
	:return: valid if in rectangle
	"""
	vertex4 = np.asarray(vertex4)
	valid_1 = in_triangle(points, vertex4[[0, 1, 2]])
	valid_2 = in_triangle(points, vertex4[[2, 3, 0]])
	return np.logical_and(valid_1, valid_2)


def intersect_line2rect(p0, p1, c0, c1):
	"""
	:param p0:
	:param p1:
	:param c0: left bottom corner
	:param c1: right top corner
	:return:
	"""
	edge = np.asarray([[0, 1, 0], [0, 1, -1],
	                   [1, 0, 0], [1, 0, -1]], np.float64)
	w, h = float(c1[0] - c0[0]), float(c1[1] - c0[1])
	p0 = np.asarray(p0, np.float64) - c0
	p1 = np.asarray(p1, np.float64) - c0
	v = p1 - p0
	line = np.asarray([v[1] / h, -v[0] / w, np.cross(v, p0) / w / h])
	temp = [np.cross(line, e) for e in edge]
	points = np.asarray([[p[0] / p[2], p[1] / p[2]] for p in temp])
	points = np.unique(points, axis = 0)
	valid = np.logical_and(points >= 0, points <= 1)
	valid = np.logical_and(valid[:, 0], valid[:, 1])
	if np.sum(valid) != 2: return None
	return points[valid, :] * [[w, h], [w, h]] + c0


def triangle_centroid(p0, p1, p2):
	x = (p0[0] + p1[0] + p2[0]) / 3
	y = (p0[1] + p1[1] + p2[1]) / 3
	return np.asarray((x, y), np.float64)


def triangle_circumcenter(p0, p1, p2):
	b0 = p0[0] * p0[0] + p0[1] * p0[1]
	b1 = p1[0] * p1[0] + p1[1] * p1[1]
	b2 = p2[0] * p2[0] + p2[1] * p2[1]
	den = p0[0] * p2[1] + p1[0] * p0[1] + p2[0] * p1[1]
	den -= p0[0] * p1[1] + p1[0] * p2[1] + p2[0] * p0[1]
	x = b0 * p2[1] + b1 * p0[1] + b2 * p1[1]
	x -= b0 * p1[1] + b1 * p2[1] + b2 * p0[1]
	x = x / den / 2
	y = p0[0] * b2 + p1[0] * b0 + p2[0] * b1
	y -= p0[0] * b1 + p1[0] * b2 + p2[0] * b0
	y = y / den / 2
	return np.asarray((x, y), np.float64)


def triangle_incenter(p0, p1, p2):
	v01 = (p1[0] - p0[0], p1[1] - p0[1])
	v02 = (p2[0] - p0[0], p2[1] - p0[1])
	v12 = (p2[0] - p1[0], p2[1] - p1[1])
	l0 = math.sqrt(v12[0] * v12[0] + v12[1] * v12[1])
	l1 = math.sqrt(v02[0] * v02[0] + v02[1] * v02[1])
	l2 = math.sqrt(v01[0] * v01[0] + v01[1] * v01[1])
	den = l0 + l1 + l2
	x = (l0 * p0[0] + l1 * p1[0] + l2 * p2[0]) / den
	y = (l0 * p0[1] + l1 * p1[1] + l2 * p2[1]) / den
	return np.asarray((x, y), np.float64)


def triangle_orthocenter(p0, p1, p2):
	v1 = [p2[0] - p1[0], p2[1] - p1[1]]
	v2 = [p2[0] - p0[0], p2[1] - p0[1]]
	b1 = v1[0] * p0[0] + v1[1] * p0[1]
	b2 = v2[0] * p1[0] + v2[1] * p1[1]
	den = v1[0] * v2[1] - v1[1] * v2[0]
	x = (b1 * v2[1] - b2 * v2[0]) / den
	y = (v1[0] * b1 - v1[1] * b2) / den
	return np.asarray((x, y), np.float64)


def min_bound_circle(points):
	"""
	Smallest-circle problem
	solved by 'Welzl's algorithm'
	https://en.wikipedia.org/wiki/Smallest-circle_problem
	:param points:
	:return:
	"""
	total = len(points)
	C = (points[1] + points[0]) / 2
	R2 = np.sum([xi * xi for xi in C - points[0]])
	for i in range(2, total):
		vec = points[i] - C
		r2 = np.dot(vec, vec)
		if r2 <= R2: continue
		C = (points[i] + points[0]) / 2
		R2 = np.sum([xi * xi for xi in C - points[0]])
		for j in range(1, i):
			vec = points[j] - C
			r2 = np.dot(vec, vec)
			if r2 <= R2: continue
			C = triangle_circumcenter(points[i], points[j], points[0])
			R2 = np.sum([xi * xi for xi in C - points[0]])
			for k in range(1, j):
				vec = points[k] - C
				r2 = np.dot(vec, vec)
				if r2 <= R2: continue
				C = triangle_circumcenter(points[i], points[j], points[k])
				R2 = np.sum([xi * xi for xi in C - points[k]])
	return C, math.sqrt(R2)


def euler_spiral_length(K1, K2, S, n: int = 2):
	"""
	K = 1 / R
	K = π * L * a^2
	S = L2 - L1
	a = sqrt((K2 - K1) / S / π)
	A = (K2 + K1) * S / 2
	:param K1: could be zero
	:param K2:
	:param S: ΔL
	:param n:
	:return:
		P: points in curve
		A: ΔΘ
	"""
	a = math.sqrt((K2 - K1) / S / math.pi)
	L1_ = K1 / a / math.pi
	L2_ = K2 / a / math.pi
	T = np.linspace(L1_, L2_, max(n, 2))
	P = np.asarray(fresnel(T)[::-1]).T / a
	return P, (K2 + K1) * S / 2


def euler_spiral_angle(K1, K2, A, n: int = 2):
	"""
	K = π * L * a^2
	Θ = K * L / 2
	Α = Θ2 - Θ1
	a = sqrt((K2^2 - K1^2) / 2 / π / A)
	S = 2A / (K2 + K1)
	:param K1: bigger R
	:param K2:
	:param A:
	:param n:
	:return:
		P: points in curve
		S: ΔL
	"""
	a = math.sqrt((K2 * K2 - K1 * K1) / 2 / math.pi / A)
	L1_ = K1 / a / math.pi
	L2_ = K2 / a / math.pi
	T = np.linspace(L1_, L2_, max(n, 2))
	P = np.asarray(fresnel(T)[::-1]).T / a
	return P, A * 2 / (K2 + K1)


if __name__ == '__main__':
	from Core.Visualization import KaisCanvas


	def main():
		vertex3 = [
			[3, -1],
			[0, 5],
			[-2, -2]
		]
		pts = np.random.random((100, 2)) * 4 - 2
		valid = in_triangle(pts, vertex3)
		invalid = np.logical_not(valid)

		can = KaisCanvas()

		can.draw_polyline([vertex3[0], vertex3[1], vertex3[2], vertex3[0]])
		can.draw_points(pts[valid], color = "green")
		can.draw_points(pts[invalid], color = "red")

		can.set_axis()
		can.show()
		can.close()
		pass


	main()
	pass
