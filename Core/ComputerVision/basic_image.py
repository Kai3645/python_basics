import math

import cv2
import numpy as np

from Core.Basic import KaisLog

log = KaisLog.get_log()


def best_resize_shape(w_src: int, h_src: int, a_dst: int, *, is_width: bool = True):
	"""
	respond to demand: keep image aspect while reshaping
	given image[:w_src, :h_src]
	reshape image[:w_fit, :h_fit] --into--> image[:w_dst, :h_dst]
	:param w_src:
	:param h_src:
	:param a_dst:
	:param is_width:
	:return:
	"""
	aspect = w_src / h_src
	if is_width:
		w_dst = a_dst
		h_dst = int(a_dst / aspect)
	else:
		h_dst = a_dst
		w_dst = int(a_dst * aspect)
	rate = min(w_src / w_dst, h_src / h_dst)
	w_fit = int(round(w_dst * rate))
	h_fit = int(round(h_dst * rate))

	err = w_fit / h_fit - w_dst / h_dst
	log.debug(f"({w_src}, {h_src}) -> ({w_fit}, {h_fit}) -> ({w_dst}, {h_dst}) -> ε = {err:.3e}")
	return w_fit, h_fit, w_dst, h_dst


def unsharp_masking(img, kernel_size = (5, 5), sigma = 1.0, amount = 1.0, threshold = 0):
	"""Return a sharpened version of the image, using an unsharp mask."""

	blurred = cv2.GaussianBlur(img, kernel_size, sigma)
	sharpened = float(amount + 1) * img - float(amount) * blurred
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	if threshold > 0:
		low_contrast_mask = np.absolute(img - blurred) < threshold
		np.copyto(sharpened, img, where = low_contrast_mask)
	return sharpened


def gray2binary(gray):
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary


def gray2binary_SlideWindow(gray, slide = 20):
	window = int(slide * 10)
	w_src, h_src = gray.shape
	x = int((w_src - window) / slide) + 1
	borderWidth_w = int((x * slide + window - w_src) / 2)
	y = int((h_src - window) / slide) + 1
	borderWidth_h = int((y * slide + window - h_src) / 2)

	gray = cv2.copyMakeBorder(
		gray,
		borderWidth_w, borderWidth_w,
		borderWidth_h, borderWidth_h,
		cv2.BORDER_DEFAULT
	)
	w, h = gray.shape
	dst = np.zeros((w, h))
	flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
	h0 = 0
	for _ in range(y):
		h1 = h0 + window
		w0 = 0

		for _ in range(x):
			w1 = w0 + window
			_, binary = cv2.threshold(gray[w0:w1, h0:h1], 0, 255, flag)
			dst[w0:w1, h0:h1] += binary

			w0 = w0 + slide
		h0 = h0 + slide
	dst = dst[borderWidth_w:borderWidth_w + w_src, borderWidth_h:borderWidth_h + h_src]
	gray = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return gray


def hist_gray(gray, th = 0):
	vote = gray.ravel()
	index, count = np.unique(vote, return_counts = True)
	x = np.asarray(index, np.float64)
	y = np.asarray(count, np.float64)
	valid = count > th
	return x[valid], y[valid]


def mask_edge_alpha(mask, size: int, K: int = 3):
	"""
	:param mask:
	:param size:
	:param K: zoom in level
	:return:
		alpha:
	"""
	if K < 2: K = 2
	if K > 2:
		kernel = np.ones((5, 5), np.uint8)
		size //= 2
		K -= 1
	else:
		kernel = np.ones((3, 3), np.uint8)
	img_h, img_w = mask.shape[:2]
	loc_w = img_w // K
	loc_h = img_h // K
	loc_s = size // K
	mask_loc = cv2.resize(mask, (loc_w, loc_h), interpolation = cv2.INTER_AREA)
	alpha = np.zeros_like(mask_loc, np.uint8)
	for i in range(loc_s):
		t = math.cos(i / loc_s * math.pi)
		a = round(128 - 127 * t)
		alpha[mask_loc > 100] = a
		mask_loc = cv2.erode(mask_loc, kernel, borderType = cv2.BORDER_CONSTANT, borderValue = 0)
	alpha[mask_loc > 100] = 255
	alpha = cv2.resize(alpha, (img_w, img_h), interpolation = cv2.INTER_LINEAR)
	if K > 1:
		s = int(K + K + 1)
		alpha = cv2.blur(alpha, (s, s), borderType = cv2.BORDER_REPLICATE)
	alpha[mask < 100] = 0
	alpha = alpha.astype(np.float32) / 255
	return alpha


def combine_alpha(alpha_1, alpha_2):
	alpha_sum = alpha_1 + alpha_2
	valid = alpha_sum > 0
	alpha_3 = np.copy(alpha_1)
	alpha_3[valid] /= alpha_sum[valid]
	alpha_4 = np.copy(alpha_2)
	alpha_4[valid] /= alpha_sum[valid]
	return alpha_3, alpha_4
