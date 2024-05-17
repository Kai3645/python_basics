"""
	Reference: "Digital Image Processing" ISBN 978-4-903474-50-2
	x for src sampling pts
	. for new resampling pts
	E for edge pts
	(0, 0)
	O------------+-----------+
	|  .     .   |  .     .  |
	|     x      |     x     |
	|  .     .   |  .     .  |
	+------------+-----------+
	|  .     .   |  .     .  |
	|     x      |     x     |
	|  .     .   |  .     .  |
	+------------+-----------+ (h, w)

	warning, the value of img[0, 0] is stand for point at (0.5, 0.5)
	warning, Never use this functions for whole image interpolation !! use cv2.remap instead
"""

import cv2
import numpy as np


def valid_in_image(Us, w, h):
	valid_w = np.logical_and(Us[:, 0] >= 0, Us[:, 0] < w)
	valid_h = np.logical_and(Us[:, 1] >= 0, Us[:, 1] < h)
	return np.logical_and(valid_w, valid_h)


def value_limit(values):
	values = values.round()
	values[values < 0] = 0
	values[values > 255] = 255
	return values.astype(np.uint8)


def nearest_inter(src, Us):
	"""
	(0, 0)
	O------------+
	|  .     .   |
	|     x      |
	|  .     .   |
	+------------+ (h, w)
	basic point idx -> (i, j) = (int(u), int(v))
	value = img[i, j]
	========================================
	:param src: source image
	:param Us: (n, 2) array in u-v CS
	:return:
		values: (n, ) array of image value
	"""
	shape = src.shape
	if len(shape) < 3: c = 1
	else: c = shape[2]
	h, w = shape[:2]
	valid = valid_in_image(Us, w, h)
	values = np.zeros((len(Us), c), float)
	for i, in_img in enumerate(valid):
		if not in_img: continue
		u_ = int(Us[i, 0])
		v_ = int(Us[i, 1])
		values[i, :] = src[v_, u_]
	values = value_limit(values)
	if c > 1: return values
	return values.ravel()


def bilinear_inter(src, Us):
	"""
	(0, 0)
	O------------+-----------+-----------+
	|            |           |           |
	|     E      |     E     |     E     |
	|            |           |           |
	+------------+-----------+-----------+
	|            |  .     .  |           |
	|     E      |     x     |     E     |
	|            |  .     .  |           |
	+------------+-----------+-----------+
	|            |           |           |
	|     E      |     E     |     E     |
	|            |           |           |
	+------------+-----------+-----------+ (h + 2, w + 2)
	basic point idx -> (i, j) = (int(u + 0.5), int(v + 0.5))
	value = func(O----+
	             |    |
	             +----+)
	========================================
	in 1D
     |<---      t      --->|--- 1 --->|
	x_0 ----------------- x_t ------ x_1
     A                    M          B
	M = (1 - t) * A + t * B
	  = (x_1 - x_t) * A + (x_t - x_0) * B
	========================================
	in 2D
	A--------B
	|   M    |
	|        |
	C--------D
	M =   (y_1 - y_t) * { (x_1 - x_t) * A + (x_t - x_0) * B }
	    + (y_t - y_0) * { (x_1 - x_t) * C + (x_t - x_0) * D }
	========================================
	:param src: source image
	:param Us: (n, 2) array in u-v CS
	:return:
		values: (n, ) array of image value
	"""
	shape = src.shape
	if len(shape) < 3: c = 1
	else: c = shape[2]
	h, w = shape[:2]
	valid = valid_in_image(Us, w, h)
	src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	Us = Us + 0.5
	values = np.zeros((len(Us), c), float)
	for i, in_img in enumerate(valid):
		if not in_img: continue
		u0 = int(Us[i, 0])
		u1 = u0 + 1
		v0 = int(Us[i, 1])
		v1 = v0 + 1
		# ========================================
		f0 = src[v0, u0]
		f1 = src[v0, u1]
		f2 = src[v1, u0]
		f3 = src[v1, u1]
		u1t = u1 - Us[i, 0]
		ut0 = Us[i, 0] - u0
		v1t = v1 - Us[i, 1]
		vt0 = Us[i, 1] - v0
		# ========================================
		values[i, :] = u1t * v1t * f0 + ut0 * v1t * f1 + u1t * vt0 * f2 + ut0 * vt0 * f3
	values = value_limit(values)
	if c > 1: return values
	return values.ravel()


def bicubic_inter(src, Us, *, alpha = -0.75):
	"""
	Reference: book, P.170
	(0, 0)
	O------------+-----------+-----------+-----------+-----------+
	|            |           |           |           |           |
	|     E      |     E     |     E     |     E     |     E     |
	|            |           |           |           |           |
	+------------+-----------+-----------+-----------+-----------+
	|            |           |           |           |           |
	|     E      |     E     |     E     |     E     |     E     |
	|            |           |           |           |           |
	+------------+-----------+-----------+-----------+-----------+
	|            |           |  .     .  |           |           |
	|     E      |     E     |     x     |     E     |     E     |
	|            |           |  .     .  |           |           |
	+------------+-----------+-----------+-----------+-----------+
	|            |           |           |           |           |
	|     E      |     E     |     E     |     E     |     E     |
	|            |           |           |           |           |
	+------------+-----------+-----------+-----------+-----------+
	|            |           |           |           |           |
	|     E      |     E     |     E     |     E     |     E     |
	|            |           |           |           |           |
	+------------+-----------+-----------+-----------+-----------+ (h + 4, w + 4)
	basic point idx -> (i, j) = (int(u + 0.5), int(v + 0.5))
	value = func(O----+----+----+
	             |    |    |    |
	             +----+----+----+
	             |    |    |    |
	             +----+----+----+
	             |    |    |    |
	             +----+----+----+)
	========================================
	:param src: source image
	:param Us: (n, 2) array in u-v CS
	:param alpha: better in [-1.0, -0.5], opencv used -0.75
	:return:
		values: (n, ) array of image value
	"""
	shape = src.shape
	if len(shape) < 3: c = 1
	else: c = shape[2]
	h, w = shape[:2]
	valid = valid_in_image(Us, w, h)
	src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
	Us = Us + 0.5
	values = np.zeros((len(Us), c), float)
	for i, in_img in enumerate(valid):
		if not in_img: continue
		u0 = int(Us[i, 0])
		u1 = u0 + 1
		u2 = u0 + 2
		u3 = u0 + 3
		v0 = int(Us[i, 1])
		v1 = v0 + 1
		v2 = v0 + 2
		v3 = v0 + 3
		# ========================================
		u_t = Us[i, 0] - u0
		u_t2 = u_t * u_t
		u_t_ = u1 - Us[i, 0]
		u_t2_ = u_t_ * u_t_
		hu_0 = alpha * u_t * u_t2_
		hu_1 = (alpha + 2) * u_t2 * u_t - (alpha + 3) * u_t2 + 1
		hu_2 = (alpha + 2) * u_t2_ * u_t_ - (alpha + 3) * u_t2_ + 1
		hu_3 = alpha * u_t_ * u_t2
		# ========================================
		v_t = Us[i, 1] - v0
		v_t2 = v_t * v_t
		v_t_ = v1 - Us[i, 1]
		v_t2_ = v_t_ * v_t_
		hv_0 = alpha * v_t * v_t2_
		hv_1 = (alpha + 2) * v_t2 * v_t - (alpha + 3) * v_t2 + 1
		hv_2 = (alpha + 2) * v_t2_ * v_t_ - (alpha + 3) * v_t2_ + 1
		hv_3 = alpha * v_t_ * v_t2
		# ========================================
		values[i, :] = (src[v0, u0] * hu_0 + src[v0, u1] * hu_1 + src[v0, u2] * hu_2 + src[v0, u3] * hu_3) * hv_0 + \
		               (src[v1, u0] * hu_0 + src[v1, u1] * hu_1 + src[v1, u2] * hu_2 + src[v1, u3] * hu_3) * hv_1 + \
		               (src[v2, u0] * hu_0 + src[v2, u1] * hu_1 + src[v2, u2] * hu_2 + src[v2, u3] * hu_3) * hv_2 + \
		               (src[v3, u0] * hu_0 + src[v3, u1] * hu_1 + src[v3, u2] * hu_2 + src[v3, u3] * hu_3) * hv_3
	values = value_limit(values)
	if c > 1: return values
	return values.ravel()


if __name__ == '__main__':
	from Core.Visualization import KaisCanvas

	can = KaisCanvas()


	def main():
		quarter = 10
		quarter2 = int(quarter * 2)
		half = int(quarter * 4)
		size = int(half * 2)
		size2 = int(size * 2)
		src = np.round(np.random.random((half, half, 3)) * 255).astype(np.uint8)
		src[quarter2:half, :quarter2] = 255
		src[quarter2:quarter2 + quarter, :quarter] = 0

		img = np.zeros((size2, size2, 3), np.uint8)
		img[:half, :half, :] = src

		ys, xs = np.mgrid[:size, :size]
		xs = xs.ravel()
		ys = ys.ravel()
		Us = np.asarray([xs, ys], float).T
		Us = (Us + 0.5) / 2

		# values = nearest_inter(src, Us)
		# values = bilinear_inter(src, Us)
		values = bicubic_inter(src, Us, alpha = -0.75)
		img[ys, xs + size, :] = values
		img[size:, :size, :] = cv2.resize(src, (size, size), interpolation = cv2.INTER_CUBIC)
		diff = cv2.absdiff(img[:size, size:], img[size:, :size])
		img[size:, size:, :] = cv2.normalize(diff, None, 0, 100, cv2.NORM_MINMAX)
		can.ax.imshow(img)
		can.show()

		pass


	main()
	can.close()
	pass
