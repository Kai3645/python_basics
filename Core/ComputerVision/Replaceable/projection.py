"""
camera forward axis = z-axis (0, 0, 1)
in this file, U = (u, v) is based on origin point (0, 0, z)
"""

import math

import numpy as np


def project_flat(P, focal, aspect = 1):
	"""
	require: (u, v) = Func(x, y, z)
	solve:
		u = focal * x / z
		v = focal * y / z
	:param P: (n, 3) array
	:param focal: distance to flat
	:param aspect: focal_x / focal_y
	:return:
		U: (n, 2) array
	"""
	focal_y = focal / aspect
	u = P[:, 0] / P[:, 2] * focal
	v = P[:, 1] / P[:, 2] * focal_y
	return np.asarray([u, v], P.dtype).T


def recover_flat(U, focal, aspect = 1):
	"""
	require: (x, y, z) = Func(u, v)
	solve:
		x = u
		y = v
		y = focal
	:param U: (n, 2) array, be sure U has been translated to center
	:param focal: distance to flat
	:param aspect: focal_x / focal_y
	:return:
		P: (n, 3) array
	"""
	P = np.zeros((len(U), 3), U.dtype)
	P[:, 0] = U[:, 0]
	P[:, 1] = U[:, 1]
	P[:, 2] = focal
	if aspect != 1: P[:, 1] *= aspect
	return P


def project_cylinder(P, focal, aspect = 1):
	"""
	require: (u, v) = Func(x, y, z)
	solve:
		da = atan2(1, focal)
		u = atan2(x, z) / da
		v = focal / sqrt(x^2 + z^2) * y
	:param P: (n, 3) array
	:param focal: distance to cylinder
	:param aspect: focal_x / focal_y
	:return:
		U: (n, 2) array
	"""
	focal_y = focal / aspect
	da = math.atan2(1, focal)
	x = P[:, 0]
	y = P[:, 1]
	z = P[:, 2]
	u = np.arctan2(x, z) / da
	v = focal_y / np.sqrt(x * x + z * z) * y
	return np.asarray([u, v], P.dtype).T


def recover_cylinder(U, focal, aspect = 1):
	"""
	require: (x, y, z) = Func(u, v)
	solve:
		da = atan2(1, focal)
		a = da * u
		x = sin(a)
		y = v / focal
		z = cos(a)
	:param U: (n, 2) array, be sure U has been translated to center
	:param focal: distance to cylinder
	:param aspect: focal_x / focal_y
	:return:
		P: (n, 3) array
	"""
	da = math.atan2(1, focal)
	a = U[:, 0] * da
	x = np.sin(a) * focal
	y = U[:, 1] * aspect
	z = np.cos(a) * focal
	if aspect != 1: y *= aspect
	return np.asarray([x, y, z], U.dtype).T


def project_stereographic(P, focal, aspect = 1):
	"""
	require: (u, v) = Func(x, y, z)
	normalization:
		r^2 = x^2 + y^2 + z^2
		x' = focal / r * x
		y' = focal / r * y
		z' = focal / r * z
		u = 2 * focal / (focal + z') * x'
		v = 2 * focal / (focal + z') * y'
	solve:
		u = 2 * focal * x / (r + z)
		v = 2 * focal * y / (r + z)
	:param P: (n, 3) array
	:param focal: distance to flat
	:param aspect: focal_x / focal_y
	:return:
		U: (n, 2) array
	"""
	focal2_x = focal * 2
	focal2_y = focal2_x / aspect
	x = P[:, 0]
	y = P[:, 1]
	z = P[:, 2]
	den = np.sqrt(x * x + y * y + z * z) + z
	u = focal2_x * x / den
	v = focal2_y * y / den
	return np.asarray([u, v], P.dtype).T


def recover_stereographic(U, focal, aspect = 1):
	"""
	require: (x, y, z) = Func(u, v)
	step: unknown ..
	solve:
		r^2 = u^2 + v^2
		k = r^2 / 4 / f^2
		x = 1 / (1 + k) * u
		y = 1 / (1 + k) * v
		z = (1 - k) / (1 + k) * f
	:param U: (n, 2) array, be sure U has been translated to center
	:param focal: distance to flat
	:param aspect: focal_x / focal_y
	:return:
		P: (n, 3) array
	"""
	u = U[:, 0]
	v = U[:, 1]
	if aspect != 1: v *= aspect
	r2 = u * u + v * v
	k = r2 / 4 / focal / focal
	t = 1 / (1 + k)
	x = t * u
	y = t * v
	z = (1 - k) * t * focal
	return np.asarray([x, y, z], U.dtype).T


if __name__ == '__main__':
	from Core.Visualization import SpaceCanvas, KaisColor

	canvas3d = SpaceCanvas("/home/kai/PycharmProjects/pyCenter/M23_03/D2303_24/out")


	def main():
		s = 10
		w, h = 40, 30
		focal = w * s
		xs, ys = np.mgrid[:int(w * 2), :int(h * 2)]
		xs = xs.ravel() - w
		ys = ys.ravel() - h
		Us_org = np.transpose([xs, ys]) * s
		pts_org = np.zeros((int(w * h * 4), 3))
		pts_org[:, :2] = Us_org
		pts_org[:, 2] = focal
		R_org = np.linalg.norm(pts_org, axis = 1)
		pts_org[:, 0] *= focal / R_org
		pts_org[:, 1] *= focal / R_org
		pts_org[:, 2] *= focal / R_org

		canvas3d.add_axis("axis", r = 1, scales = (focal, focal, focal))
		canvas3d.add_point("org", pts_org, color = KaisColor.plotColor("white"))

		new_focal = focal
		# Us = project_flat(pts_org, new_focal)
		# Us = project_cylinder(pts_org, new_focal)
		Us = project_stereographic(pts_org, new_focal)
		pts = np.zeros_like(pts_org)
		pts[:, :2] = Us
		pts[:, 2] = new_focal
		canvas3d.add_point("new", pts, color = KaisColor.plotColor("yellow"))

		# pts = recover_flat(Us, new_focal)
		pts = recover_cylinder(Us, new_focal)
		# pts = recover_stereographic(Us, new_focal)
		canvas3d.add_point("new", pts, color = KaisColor.plotColor("lime"))

		pass


	main()
	canvas3d.save()
	pass
