import cv2
import numpy as np


def average_imageSum(image_sum, alpha_sum):
	h, w = image_sum.shape[:2]
	valid = alpha_sum > 0
	dst = np.zeros((h, w, 3), np.uint8)
	dst[valid, 0] = np.round(image_sum[valid, 0] / alpha_sum[valid])
	dst[valid, 1] = np.round(image_sum[valid, 1] / alpha_sum[valid])
	dst[valid, 2] = np.round(image_sum[valid, 2] / alpha_sum[valid])
	return dst


def sample_imageSum(image_sum, alpha_sum, roi_sum, roi_loc):
	w0 = int(roi_loc[0] - roi_sum[0] + 0.5)
	h0 = int(roi_loc[1] - roi_sum[1] + 0.5)
	w1 = int(roi_loc[2] - roi_sum[0] + 0.5)
	h1 = int(roi_loc[3] - roi_sum[1] + 0.5)
	image_loc = image_sum[h0:h1, w0:w1]
	alpha_loc = alpha_sum[h0:h1, w0:w1]

	dst = np.zeros_like(image_loc, np.uint8)
	mask = np.zeros_like(alpha_loc, np.uint8)

	valid = alpha_loc > 0
	dst[valid, 0] = np.round(image_loc[valid, 0] / alpha_loc[valid])
	dst[valid, 1] = np.round(image_loc[valid, 1] / alpha_loc[valid])
	dst[valid, 2] = np.round(image_loc[valid, 2] / alpha_loc[valid])
	mask[valid] = 255
	return dst, mask


def add_weighted(image_sum, alpha_sum, roi_sum, image_src, alpha_src, roi_src, cancel_diff = False):
	u0 = min(roi_sum[0], roi_src[0])
	v0 = min(roi_sum[1], roi_src[1])
	u1 = max(roi_sum[2], roi_src[2])
	v1 = max(roi_sum[3], roi_src[3])
	new_W = int(u1 - u0 + 0.5)
	new_H = int(v1 - v0 + 0.5)
	new_roi_sum = (u0, v0, u0 + new_W, v0 + new_H)
	new_image_sum = np.zeros((new_H, new_W, 3))
	new_alpha_sum = np.zeros((new_H, new_W))

	tmp_h, tmp_w = alpha_sum.shape
	w0 = int(roi_sum[0] - u0 + 0.5)
	h0 = int(roi_sum[1] - v0 + 0.5)
	w1 = w0 + tmp_w
	h1 = h0 + tmp_h
	new_image_sum[h0:h1, w0:w1, :] = image_sum
	new_alpha_sum[h0:h1, w0:w1] = alpha_sum

	tmp_h, tmp_w = alpha_src.shape
	w0 = int(roi_src[0] - u0 + 0.5)
	h0 = int(roi_src[1] - v0 + 0.5)
	w1 = w0 + tmp_w
	h1 = h0 + tmp_h

	if cancel_diff:
		sub_image_sum = new_image_sum[h0:h1, w0:w1]
		sub_alpha_sum = new_alpha_sum[h0:h1, w0:w1]
		image_sub = average_imageSum(sub_image_sum, sub_alpha_sum)
		loc_w = w1 - w0
		loc_h = h1 - h0
		mask_sub = np.zeros((loc_h, loc_w), np.uint8)
		# common area
		mask_sub[np.logical_and(sub_alpha_sum > 0, alpha_src > 0)] = 255
		# different area
		gray_sub = cv2.cvtColor(image_sub, cv2.COLOR_BGR2GRAY)
		gray_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
		# different in common area
		diff = cv2.absdiff(gray_src, gray_sub)
		mask_sub[diff < 50] = 0
		kernel = np.ones((3, 3), np.uint8)
		mask_sub = cv2.dilate(mask_sub, kernel, iterations = 2)

		valid = mask_sub > 100
		sub_image_sum[valid, 0] *= 0.996
		sub_image_sum[valid, 1] *= 0.996
		sub_image_sum[valid, 2] *= 0.996
		sub_alpha_sum[valid] *= 0.996
		alpha_src[valid] *= 0.667
		new_image_sum[h0:h1, w0:w1] = sub_image_sum
		new_alpha_sum[h0:h1, w0:w1] = sub_alpha_sum

	new_image_sum[h0:h1, w0:w1, 0] += image_src[:, :, 0] * alpha_src
	new_image_sum[h0:h1, w0:w1, 1] += image_src[:, :, 1] * alpha_src
	new_image_sum[h0:h1, w0:w1, 2] += image_src[:, :, 2] * alpha_src
	new_alpha_sum[h0:h1, w0:w1] += alpha_src
	return new_image_sum, new_alpha_sum, new_roi_sum


def sum_weighted(image_list, alpha_list, roi_list):
	roi_list = np.asarray(roi_list, np.float32)
	u0 = np.min(roi_list[:, 0])
	v0 = np.min(roi_list[:, 1])
	u1 = np.max(roi_list[:, 2])
	v1 = np.max(roi_list[:, 3])
	sum_W = int(u1 - u0 + 0.5)
	sum_H = int(v1 - v0 + 0.5)
	roi_sum = (u0, v0, u0 + sum_W, v0 + sum_H)

	image_sum = np.zeros((sum_H, sum_W, 3))
	alpha_sum = np.zeros((sum_H, sum_W))
	for i, roi in enumerate(roi_list):
		tmp_h, tmp_w = alpha_list[i].shape
		w0 = int(roi[0] - u0 + 0.5)
		h0 = int(roi[1] - v0 + 0.5)
		w1 = w0 + tmp_w
		h1 = h0 + tmp_h
		image_sum[h0:h1, w0:w1, 0] += image_list[i][:, :, 0] * alpha_list[i]
		image_sum[h0:h1, w0:w1, 1] += image_list[i][:, :, 1] * alpha_list[i]
		image_sum[h0:h1, w0:w1, 2] += image_list[i][:, :, 2] * alpha_list[i]
		alpha_sum[h0:h1, w0:w1] += alpha_list[i]
	return image_sum, alpha_sum, roi_sum
