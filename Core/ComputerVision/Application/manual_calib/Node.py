import math
from datetime import datetime
from time import sleep

import cv2
import numpy as np


def circle_mask(W, H, K = 1.0):
	a_ww, a_hh = np.meshgrid(np.arange(W), np.arange(H))
	pts = np.vstack([a_hh.ravel(), a_ww.ravel()]).T
	mask = np.zeros((H, W), dtype = np.uint8)
	r = min(W / 2, H / 2) * K
	d = np.linalg.norm(pts + 0.5 - (H / 2, W / 2), axis = 1)
	xs, ys = pts[d < r].T
	mask[xs, ys] = 1
	return mask


def hist(X, th, N = 20):
	center = np.average(X)
	valid = None
	for i in range(N):
		valid = np.abs(X - center) < th
		if np.sum(valid) == 0:
			return center, np.logical_not(valid)
		c_tmp = np.average(X[valid])
		if abs(c_tmp - center) < 0.1: break
		center = c_tmp
	return center, valid


def detect_center_cross(image, mask, th):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3, 3), 2)
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	img[mask == 0] = 0
	
	kernel = np.ones((3, 3), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 1)
	img = cv2.erode(img, kernel, iterations = 1)
	
	ys, xs = np.argwhere(img > 200).T
	pts = np.vstack([xs, ys]).T
	
	h, w = img.shape
	
	if len(pts) < 16: return w / 2, h / 2
	
	best_idx = []
	distance = []
	count_in = 0
	for i, pt in enumerate(pts):
		D = np.min(np.abs(pts - pt), axis = 1)
		valid = D < th
		count = np.sum(valid)
		di = np.average(D[valid])
		
		if count == count_in:
			best_idx.append(i)
			distance.append(di)
		elif count > count_in:
			best_idx = [i]
			distance = [di]
			count_in = count
	cx0, cy0 = pts[best_idx[np.argmin(distance)]]
	cx1 = np.average(xs[np.logical_and(xs > cx0 - th, xs < cx0 + th)])
	cy1 = np.average(ys[np.logical_and(ys > cy0 - th, ys < cy0 + th)])
	# if (cx1 - cx0) * (cx1 - cx0) > 4 or (cy1 - cy0) * (cy1 - cy0) > 4:
	# 	return float(cy0), float(cy0)
	return float(cx1), float(cy1)


class CNode:
	def __init__(self, pos):
		self.pos = [pos[0], pos[1]]
		self.N0 = None
		self.N1 = None
		self.N2 = None
		self.N3 = None
		pass
	
	def set_sub(self, N0, N1, N2, N3):
		self.N0 = N0
		self.N1 = N1
		self.N2 = N2
		self.N3 = N3
		pass
	
	def get_pos(self):
		return float(self.pos[0]), float(self.pos[1])
	
	def set_pos(self, x, y):
		self.pos[:] = x, y
	
	def get_corners(self):
		P0 = self.N0.get_pos()
		P1 = self.N1.get_pos()
		P2 = self.N2.get_pos()
		P3 = self.N3.get_pos()
		return np.asarray([P0, P1, P2, P3], float)
	
	# def get_roi(self):
	# 	pts = self.get_corners()
	# 	h0 = int(np.min(pts[:, 0]))
	# 	w0 = int(np.min(pts[:, 1]))
	# 	h1 = math.ceil(np.max(pts[:, 0]))
	# 	w1 = math.ceil(np.max(pts[:, 1]))
	# 	return h0, w0, h1, w1
	
	def detect(self, W, H, image, mask, th):
		pts = self.get_corners()
		x0 = int(np.min(pts[:, 0]))
		y0 = int(np.min(pts[:, 1]))
		x1 = math.ceil(np.max(pts[:, 0])) + 1
		y1 = math.ceil(np.max(pts[:, 1])) + 1
		
		P_src = np.asarray(pts - (x0, y0), np.float32)
		P_dst = np.asarray([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
		mat = cv2.getPerspectiveTransform(P_src, P_dst)
		dst = cv2.warpPerspective(image[y0:y1, x0:x1], mat, (W, H), flags = cv2.INTER_CUBIC,
		                          borderMode = cv2.BORDER_REFLECT)
		C = detect_center_cross(dst, mask, th = th)
		
		t_current = datetime.now()
		s = t_current.strftime("%Y_%m%d_%H%M_%S.%f")
		folder = "/home/lab/Desktop/python_resource/M24_12/D2412_28/out/"
		tmp = np.copy(image[y0:y1, x0:x1])
		tmp = cv2.polylines(tmp, np.array([P_src.round()], np.int32), isClosed = True, color = (255, 0, 0),
		                    thickness = 1)
		cv2.imwrite(folder + "img_" + s + "a.jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
		dst = cv2.circle(dst, (round(C[0]), round(C[1])), 5, (255, 80, 0), cv2.FILLED)
		cv2.imwrite(folder + "img_" + s + "b.jpg", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
		sleep(0.01)
		
		inv_mat = np.linalg.inv(mat)
		den = inv_mat[2, 0] * C[0] + inv_mat[2, 1] * C[1] + inv_mat[2, 2]
		cx = float((inv_mat[0, 0] * C[0] + inv_mat[0, 1] * C[1] + inv_mat[0, 2]) / den)
		cy = float((inv_mat[1, 0] * C[0] + inv_mat[1, 1] * C[1] + inv_mat[1, 2]) / den)
		k1 = min((x1 - x0 - cx), cx) / max((x1 - x0 - cx), cx)
		k2 = min((y1 - y0 - cy), cy) / max((y1 - y0 - cy), cy)
		if k1 < 0.2 or k2 < 0.2: cx, cy = np.mean(P_src, axis = 0)
		tmp = cv2.circle(tmp, (round(cx), round(cy)), 5, (255, 80, 0), cv2.FILLED)
		cv2.imwrite(folder + "img_" + s + "c.jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
		return float(cx + x0), float(cy + y0)
