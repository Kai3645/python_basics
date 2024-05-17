"""
camera projection -> memo: not equal to flat projection
-----------------------------------------------
(x, y, z) -> (u, v)
-----------------------------------------------
| fx    -   cx |   | x / z |   | u |
|  -   fy   cy | * | y / z | = | v |
|  -    -    1 |   |   1   |   | 1 |
memo: in camera, fy -> fx * k, k means the ratio of pixel's w-h
===============================================
(u, v) -> (x, y, z)
-----------------------------------------------
| 1 / fx     0      -cx / fx |   | u |   | x |
|   0      1 / fy   -cy / fy | * | v | = | y |
|   0        0          1    |   | 1 |   | z |
===============================================
in this file, U = (u_, v_) should be normalized
	u_ = (u - cx) / fx
	v_ = (v - cy) / fy
"""

import numpy as np

from Core.Basic import KaisLog
from Core.Math import gaussian_elimination

log = KaisLog.get_log()


def camera_normalize(Us, fx, fy, cx, cy):
	"""
	u_ = (u - cx) / fx
	v_ = (v - cy) / fy
	:param Us: (n, 2) array
	:param fx:
	:param fy:
	:param cx:
	:param cy:
	:return:
		norm_Us: (n, 2) array
	"""
	Us_ = np.copy(Us)
	Us_[:, 0] = (Us_[:, 0] - cx) / fx
	Us_[:, 1] = (Us_[:, 1] - cy) / fy
	return Us_


def camera_recover(Us, fx, fy, cx, cy):
	"""
	u = u * fx +  cx
	v = v * fy +  cy
	:param Us: (n, 2) array
	:param fx:
	:param fy:
	:param cx:
	:param cy:
	:return:
		Us: (n, 2) array
	"""
	Us_ = np.copy(Us)
	Us_[:, 0] = Us_[:, 0] * fx + cx
	Us_[:, 1] = Us_[:, 1] * fy + cy
	return Us_


def getPerspectiveMat(U_src, U_dst):
	"""
	require: U_dst = func(U_src)
	----------------------------------------
	key tech: homography, flat to flat
	t(u', v', 1) = H * t(u, v, 1)
	projection perspective:
	| u' |    | h0  h1  h2 |   | u |
	| v' |  = | h3  h4  h5 | * | v |
	| 1  |    | h6  h7   1 |   | 1 |
	----------------------------------------
	| u1  v1  1             -u1.u1'  -v1.u1' |   | h0 |   |  u1' |
	|            u1  v1  1  -u1.v1'  -v1.v1' |   | h1 |   |  v1' |
	| u2  v2  1             -u2.u2'  -v2.u2' |   | h2 |   |  u2' |
	|            u2  v2  1  -u2.v2'  -v2.v2' | * | h3 | = |  v2' |
	|     :          :         :        :    |   | h4 |   |   :  |
	|     :          :         :        :    |   | h5 |   |   :  |
	| ui  vi  1             -ui.ui'  -vi.ui' |   | h6 |   |  ui' |
	|            ui  vi  1  -ui.vi'  -vi.vi' |   | h7 |   |  vi' |
	----------------------------------------
	replaceable:
	mat = cv2.getPerspectiveTransform(U_src, U_dst)
	:param U_src: (n, 2) array, should be normalized
	:param U_dst: (n, 2) array, should be normalized
	:return:
		mat: (3, 3) array
	"""
	N = len(U_src)
	assert N == 4, log.error(f"get_perspective_matrix, pts count = {N} != 4 ..")
	UU_ = U_src[:4, 0] * U_dst[:4, 0]
	UV_ = U_src[:4, 0] * U_dst[:4, 1]
	VU_ = U_src[:4, 1] * U_dst[:4, 0]
	VV_ = U_src[:4, 1] * U_dst[:4, 1]
	M = np.asarray([
		[U_src[0, 0], U_src[0, 1], 1, 0, 0, 0, -UU_[0], -VU_[0], U_dst[0, 0]],
		[0, 0, 0, U_src[0, 0], U_src[0, 1], 1, -UV_[0], -VV_[0], U_dst[0, 1]],
		[U_src[1, 0], U_src[1, 1], 1, 0, 0, 0, -UU_[1], -VU_[1], U_dst[1, 0]],
		[0, 0, 0, U_src[1, 0], U_src[1, 1], 1, -UV_[1], -VV_[1], U_dst[1, 1]],
		[U_src[2, 0], U_src[2, 1], 1, 0, 0, 0, -UU_[2], -VU_[2], U_dst[2, 0]],
		[0, 0, 0, U_src[2, 0], U_src[2, 1], 1, -UV_[2], -VV_[2], U_dst[2, 1]],
		[U_src[3, 0], U_src[3, 1], 1, 0, 0, 0, -UU_[3], -VU_[3], U_dst[3, 0]],
		[0, 0, 0, U_src[3, 0], U_src[3, 1], 1, -UV_[3], -VV_[3], U_dst[3, 1]],
	], UU_.dtype)
	X, dof, _ = gaussian_elimination(M)
	if dof > 0: log.warning("getPerspectiveMat, DOF > 0, something may wrong ..")
	return np.asarray([[X[0], X[1], X[2]], [X[3], X[4], X[5]], [X[6], X[7], 1]], UU_.dtype)


def perspective(H, U_src):
	"""
	set with get_perspective_matrix
	----------------------------------------
	U_src -> (u, v)
	U_dst -> (u_, v_)
	projection perspective:
	t(u, v, 1) = H * t(u', v', 1)
	----------------------------------------
	H = | h00  h01  h02 |
	    | h10  h11  h12 |
	    | h20  h21  h22 |
	den = h20 * u + h21 * v + h22
	u_ =  (h00 * u + h01 * v + h02) / den
	v_ =  (h10 * u + h11 * v + h12) / den
	----------------------------------------
	:param H: (3, 3) array
	:param U_src: (n, 2) array
	:return:
		U_dst: (n, 2) array
	"""
	den = H[2, 0] * U_src[:, 0] + H[2, 1] * U_src[:, 1] + H[2, 2]
	u_ = (H[0, 0] * U_src[:, 0] + H[0, 1] * U_src[:, 1] + H[0, 2]) / den
	v_ = (H[1, 0] * U_src[:, 0] + H[1, 1] * U_src[:, 1] + H[1, 2]) / den
	return np.asarray([u_, v_], np.float64).transpose()


if __name__ == '__main__':
	import cv2


	def main():
		Image_W, Image_H = 1920, 1080

		H_ = np.random.random((3, 3))
		H_[2, 2] = 1

		corners = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], np.float64) * (Image_W, Image_H)
		corners_dst = perspective(H_, corners)

		H = getPerspectiveMat(corners, corners_dst)
		H_cv = cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_dst))
		print("H diff = ")
		print(H - H_cv)
		print()

		corners_dst_ = perspective(H, corners)
		print("my pts diff = ")
		print(corners_dst - corners_dst_)
		print()

		corners_dst_ = perspective(H_cv, corners)
		print("cv pts diff = ")
		print(corners_dst - corners_dst_)
		print()

		pass


	main()
	pass
