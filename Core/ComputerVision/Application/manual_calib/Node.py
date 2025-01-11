import math

import cv2
import numpy as np

folder = "/home/lab/Desktop/python_resource/M24_12/D2412_28/out/"


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


def detect_center_cross(img, mask, th):
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	kernel = np.ones((3, 3), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 1)
	img = cv2.erode(img, kernel, iterations = 2)
	img[mask == 0] = 0
	
	th = round(th)
	h, w = img.shape
	ys, xs = np.argwhere(img > 200).T
	
	if len(xs) < 10: return w / 2, h / 2
	
	hist_w = np.sum(img, axis = 0)
	tmp_w = np.zeros(w + 2 * th, float)
	for i in range(th * 2 + 1):
		tmp_w[i:i + w] += hist_w
	tmp_w[th:th + w] += hist_w
	cx = np.argmax(tmp_w) - th
	
	hist_h = np.sum(img, axis = 1)
	tmp_h = np.zeros(h + 2 * th, float)
	for i in range(th * 2 + 1):
		tmp_h[i:i + h] += hist_h
	tmp_h[th:th + h] += hist_h
	cy = np.argmax(tmp_h) - th
	
	th2 = th * 2
	cx = float(np.mean(xs[np.logical_and(xs > cx - th2, xs < cx + th2)]))
	cy = float(np.mean(ys[np.logical_and(ys > cy - th2, ys < cy + th2)]))
	return cx, cy


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
	
	def detect(self, W, H, image, mask, th):
		pts = self.get_corners()
		x0 = int(np.min(pts[:, 0]))
		y0 = int(np.min(pts[:, 1]))
		x1 = math.ceil(np.max(pts[:, 0])) + 1
		y1 = math.ceil(np.max(pts[:, 1])) + 1
		
		P_src = np.asarray(pts - (x0, y0), np.float32)
		P_dst = np.asarray([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
		# inv_mat = cv2.getPerspectiveTransform(P_dst, P_src)
		# mat = np.linalg.inv(inv_mat)
		mat = cv2.getPerspectiveTransform(P_src, P_dst)
		dst = cv2.warpPerspective(image[y0:y1, x0:x1], mat, (W, H),
		                          flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT)
		cx0, cy0 = detect_center_cross(dst, mask, th = th)
		
		# s = datetime.now().strftime("%Y_%m%d_%H%M_%S.%f")
		# dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
		# dst = cv2.circle(dst, (round(cx0), round(cy0)), 2, (0, 80, 255), cv2.FILLED)
		# cv2.imwrite(folder + "img_" + s + "a.jpg", dst)
		
		inv_mat = cv2.getPerspectiveTransform(P_dst, P_src)
		# inv_mat = np.linalg.inv(mat)
		den = inv_mat[2, 0] * cx0 + inv_mat[2, 1] * cy0 + inv_mat[2, 2]
		cx1 = float((inv_mat[0, 0] * cx0 + inv_mat[0, 1] * cy0 + inv_mat[0, 2]) / den)
		cy1 = float((inv_mat[1, 0] * cx0 + inv_mat[1, 1] * cy0 + inv_mat[1, 2]) / den)
		# cx1 = float((inv_mat[0, 0] * cx0 + inv_mat[0, 1] * cy0 + inv_mat[0, 2]) / den) + 0.45
		# cy1 = float((inv_mat[1, 0] * cx0 + inv_mat[1, 1] * cy0 + inv_mat[1, 2]) / den) + 0.45
		
		k1 = min((x1 - x0 - cx1), cx1) / max((x1 - x0 - cx1), cx1)
		k2 = min((y1 - y0 - cy1), cy1) / max((y1 - y0 - cy1), cy1)
		if k1 < 0.2 or k2 < 0.2: cx1, cy1 = np.mean(P_src, axis = 0)
		
		# tmp = np.copy(image[y0:y1, x0:x1])
		# tmp = cv2.polylines(tmp, np.array([P_src.round()], np.int32),
		#                     isClosed = True, color = (255, 0, 0), thickness = 1)
		# tmp = cv2.circle(tmp, (round(cx1), round(cy1)), 2, (255, 80, 0), cv2.FILLED)
		# cv2.imwrite(folder + "img_" + s + "b.jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
		# sleep(0.01)
		
		return cx1 + x0, cy1 + y0
