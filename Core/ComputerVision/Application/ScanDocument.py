from itertools import combinations

import cv2
import numpy as np

from Core.Statistic import PCA


def get_backgrounds(imgs):
	h, w = imgs[0].shape[:2]
	bgs = []

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	for img in imgs:
		src = cv2.GaussianBlur(img, (5, 5), 3)
		src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 5)

		dst = np.zeros((h, w, 4), np.uint8)
		tmp = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
		dst[:, :, 3] = tmp[:, :, 0]
		tmp[:, :, 0] = 0
		dst[:, :, :3] = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)
		bgs.append(dst)
	return bgs


def multi_canny(src):
	th1 = 0
	th2 = 128
	size = 15

	h, w = src.shape[:2]
	mask = np.zeros((h, w), np.uint8)

	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	mask[cv2.Canny(tmp, th1, th2, size) > 0] = 1

	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
	mask[cv2.Canny(tmp[:, :, 0], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 1], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 2], th1, th2, size) > 0] = 1
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	mask[cv2.Canny(tmp[:, :, 0], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 1], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 2], th1, th2, size) > 0] = 1
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
	mask[cv2.Canny(tmp[:, :, 0], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 1], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 2], th1, th2, size) > 0] = 1
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)
	mask[cv2.Canny(tmp[:, :, 0], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 1], th1, th2, size) > 0] = 1
	mask[cv2.Canny(tmp[:, :, 2], th1, th2, size) > 0] = 1
	return mask


def canny_SlideWindow(src, mask):
	h_src, w_src = mask.shape
	slide = max(min(h_src, w_src) // 39 + 1, 30)
	win = int(slide * 3)

	x = int((w_src - win) / slide) + 1
	b_w = int((x * slide + win - w_src) / 2)
	y = int((h_src - win) / slide) + 1
	b_h = int((y * slide + win - h_src) / 2)

	img = cv2.copyMakeBorder(src, b_h, b_h, b_w, b_w, cv2.BORDER_REFLECT)
	mask = cv2.copyMakeBorder(mask, b_h, b_h, b_w, b_w, cv2.BORDER_DEFAULT)
	h, w = mask.shape
	count = np.zeros((h, w))
	h0 = 0
	for _ in range(y):
		h1 = h0 + win
		w0 = 0
		for _ in range(x):
			w1 = w0 + win
			if np.sum(mask[h0:h1, w0:w1] > 0) > 0:
				can = multi_canny(img[h0:h1, w0:w1])
				count[h0:h1, w0:w1] += can
			w0 = w0 + slide
		h0 = h0 + slide
	dst = np.zeros((h, w), np.uint8)
	dst[count > 0] = 255

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	dst = cv2.dilate(dst, kernel, iterations = 3)
	return dst[b_h:b_h + h_src, b_w:b_w + w_src]


def get_edgeMask(src, bgs):
	h, w = src.shape[:2]
	k_s = max(min(w, h) // 100, 3)

	src = cv2.GaussianBlur(src, (5, 5), 3)
	kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel_5, iterations = 5)

	src_ = np.zeros((h, w, 4), np.uint8)
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	src_[:, :, 3] = tmp[:, :, 0]
	tmp[:, :, 0] = 0
	src_[:, :, :3] = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

	vote = np.zeros((h, w), int)
	vote_th = max((len(bgs) * 0.75), 2)
	for bg in bgs:
		diff = cv2.absdiff(src_, bg)
		diff[:, :, :3] //= 50
		diff[:, :, 3] //= 10
		valid = np.sum(diff, axis = 2) > 0
		vote[valid] += 1
	mask = np.zeros((h, w), np.uint8)
	mask[vote >= vote_th] = 255

	kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (k_s, k_s))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_e, iterations = 1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_e, iterations = 5)
	edge_outer = cv2.dilate(mask, kernel_5, iterations = 3)
	edge_inner = cv2.erode(mask, kernel_5, iterations = 3)
	mask = cv2.absdiff(edge_outer, edge_inner)

	retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
	label = 0
	min_area = w * h / 16
	for i in range(1, retval):
		area = stats[i][2] * stats[i][3]
		if area < min_area: continue
		min_area = area
		label = i
	if label == 0: return None, None
	mask[:, :] = 0
	mask[labels == label] = 255
	best_rect = stats[label, :4]

	edge = canny_SlideWindow(src, mask)
	mask[edge == 0] = 0
	mask = cv2.dilate(mask, kernel_5, iterations = 1)
	return mask, best_rect


def _distance_LP(p0, v0, pts):
	P = pts - p0
	K = np.tile(np.dot(P, v0), (2, 1)).transpose()
	return np.linalg.norm(P - K * v0, axis = 1)


def _line_fitting(pts, th):
	s = 50
	N = len(pts)
	if N < 2: return np.asarray([0, 0]), np.asarray([1, 0])
	if N < s:
		p, _, u = PCA(pts, ccr_th = 1)
		v = u[:, 0] / np.linalg.norm(u[:, 0])
		return p, v

	random_indexes = (np.random.random(N)).argsort()
	pts = np.asarray(pts[random_indexes[:s]], np.float64)

	idxes_pool = list(combinations(range(s), 2))

	best_inlier = 0
	best_valid = None
	for idxes in idxes_pool:
		p = pts[idxes[0]]
		v = pts[idxes[1]] - p
		v /= np.linalg.norm(v)
		Ds = _distance_LP(p, v, pts)

		valid = Ds < th
		inlier = np.sum(valid)
		if inlier > best_inlier:
			best_inlier = inlier
			best_valid = valid
	p, _, u = PCA(pts[best_valid], ccr_th = 1)
	v = u[:, 0] / np.linalg.norm(u[:, 0])
	return p, v


def edgeLine_segment(edge, rect):
	Y, X = np.where(edge > 0)
	pts = np.transpose([X, Y])
	u0, v0, du, dv = rect
	lps = np.asarray([
		[u0, v0],
		[u0 + du, v0],
		[u0 + du, v0 + dv],
		[u0, v0 + dv],
	], np.float64)
	lvs = np.asarray([
		[0, 1], [1, 0],
		[0, 1], [1, 0],
	], np.float64)

	Ds = np.zeros((len(pts), 4))
	counts_new = np.asarray((0, 0, 0, 0))

	for i in range(4):
		Ds[:, i] = _distance_LP(lps[i], lvs[i], pts)
	Class = np.argmin(Ds, axis = 1)

	min_sigma = 99
	for count in range(10):
		counts_old = np.copy(counts_new)

		for i in range(4):
			sub = pts[Class == i]
			if len(sub) < 2: continue
			lps[i], lvs[i] = _line_fitting(sub, min_sigma * 5)
			Ds[:, i] = _distance_LP(lps[i], lvs[i], pts)

			ds = Ds[Class == i, i]
			sigma = np.sqrt(np.average(ds * ds))
			if sigma < min_sigma: min_sigma = sigma
			valid = Ds[:, i] < min_sigma * 3
			Class[Class == i] = 4
			Class[valid] = i
			counts_new[i] = np.sum(valid)
		# print("flag = ", np.sum(np.abs(counts_new - counts_old)))
		if np.sum(np.abs(counts_new - counts_old)) < 10: break
	labels = np.zeros_like(edge)
	labels[pts[:, 1], pts[:, 0]] = Class + 1
	return lps, lvs, labels


def _intersect(p1, v1, p2, v2):
	v_1x2x = v1[0] * v2[0]
	v_1x2y = v1[0] * v2[1]
	v_1y2x = v1[1] * v2[0]
	v_1y2y = v1[1] * v2[1]

	den = v_1y2x - v_1x2y
	if abs(den) < 1e-6: return False, 0, 0
	x = (v_1y2x * p1[0] - v_1x2y * p2[0] - v_1x2x * (p1[1] - p2[1])) / den
	y = (v_1x2y * p1[1] - v_1y2x * p2[1] - v_1y2y * (p1[0] - p2[0])) / -den
	return True, x, y


def get_edgeCorners(lps, lvs):
	# calc 4 corners by line-line intersect
	flag, xi, yi = _intersect(lps[0], lvs[0], lps[1], lvs[1])
	if not flag: return None
	c0 = (xi, yi)
	flag, xi, yi = _intersect(lps[2], lvs[2], lps[1], lvs[1])
	if not flag: return None
	c1 = (xi, yi)
	flag, xi, yi = _intersect(lps[2], lvs[2], lps[3], lvs[3])
	if not flag: return None
	c2 = (xi, yi)
	flag, xi, yi = _intersect(lps[0], lvs[0], lps[3], lvs[3])
	if not flag: return None
	c3 = (xi, yi)
	return np.asarray([c0, c1, c2, c3], np.float32)


def _draw_corners(img, pos, s, color):
	x, y = int(pos[0]), int(pos[1])
	cv2.circle(img, (x, y), 20, color, 3)
	cv2.putText(img, s, (x + 30, y - 20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
	            3.2, (50, 30, 0), 20, cv2.LINE_AA)
	cv2.putText(img, s, (x + 30, y - 20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
	            3.2, (20, 120, 255), 3, cv2.LINE_AA)
	return img


def test_scan(src, bgs):
	dst = np.copy(src)

	edge, rect = get_edgeMask(src, bgs)
	if edge is None: return dst, None, None
	lps, lvs, labels = edgeLine_segment(edge, rect)

	edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
	corners = get_edgeCorners(lps, lvs)
	colors = np.asarray(((239, 130, 238), (0, 139, 249), (50, 204, 50), (226, 105, 65), (255, 255, 255)), np.uint8)
	for i, c in enumerate(colors, 1): edge[labels == i] = c
	if corners is None: return dst, edge, None

	for i, j in [[0, 1], [1, 2], [2, 3], [3, 0]]:
		p1 = (int(corners[i][0]), int(corners[i][1]))
		p2 = (int(corners[j][0]), int(corners[j][1]))
		cv2.line(dst, p1, p2, (255, 132, 0), 1, lineType = cv2.LINE_AA)
	_draw_corners(dst, corners[0], "A", (255, 255, 255))
	_draw_corners(dst, corners[1], "B", (0, 0, 255))
	_draw_corners(dst, corners[2], "C", (0, 255, 0))
	_draw_corners(dst, corners[3], "D", (255, 0, 0))
	_draw_corners(edge, corners[0], "A", (255, 255, 255))
	_draw_corners(edge, corners[1], "B", (255, 255, 255))
	_draw_corners(edge, corners[2], "C", (255, 255, 255))
	_draw_corners(edge, corners[3], "D", (255, 255, 255))
	edge = cv2.addWeighted(edge, 0.8, src, 0.2, 0)
	return dst, edge, corners


def warp(src, corners, W, H, E, ppi, order):
	cm2pix = ppi / 2.54
	wf = W * cm2pix
	hf = H * cm2pix
	ef = E * cm2pix
	corners_real = np.asarray([
		[0, 0],
		[wf, 0],
		[wf, hf],
		[0, hf],
	], np.float32)[order]
	corners_real = corners_real

	mat = cv2.getPerspectiveTransform(corners, corners_real)
	dst = cv2.warpPerspective(src, mat, (int(wf), int(hf)), flags = cv2.INTER_CUBIC)

	border = int(round(ef))
	tmp_w = int(wf - ef)
	tmp_h = int(hf - ef)
	dst = dst[border:tmp_h, border:tmp_w]
	dst = cv2.copyMakeBorder(dst, border, border, border, border, cv2.BORDER_REFLECT)
	return cv2.GaussianBlur(dst, (3, 3), 1.41)


def scanDoc(src, bgs, W, H, E, ppi, order):
	"""
	:param src:
	    A -->-- B
	    :       : |
	    :       v ?
	    : - ? - : |
	    D --<-- C
	:param bgs:
	:param W: [cm]
	:param H: [cm]
	:param E: [cm]
	:param ppi: [pixel/inch]
	:param order: (A, B, C, D) = ?
	    0 -->-- 1
	    :       : |
	    :       v H
	    : - W - : |
	    3 --<-- 2
	:return:
	"""
	dst, edge, corners = test_scan(src, bgs)

	if corners is None: return None, edge
	dst = warp(src, corners, W, H, E, ppi, order)
	return dst, edge


if __name__ == '__main__':
	from Core.Basic import listdir

	def get_vote(src, bgs):
		h, w = src.shape[:2]
		src = cv2.GaussianBlur(src, (5, 5), 3)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 5)

		src_ = np.zeros((h, w, 4), np.uint8)
		tmp = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
		src_[:, :, 3] = tmp[:, :, 0]
		tmp[:, :, 0] = 0
		src_[:, :, :3] = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

		vote = np.zeros((h, w), int)
		vote_th = max((len(bgs) * 0.75), 2)
		for bg in bgs:
			diff = cv2.absdiff(src_, bg)
			diff[:, :, :3] //= 50
			diff[:, :, 3] //= 10
			valid = np.sum(diff, axis = 2) > 0
			vote[valid] += 1

		tmp = cv2.normalize(vote, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		dst = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
		dst[vote > vote_th, :] = (50, 200, 10)
		return dst

	def test():
		imgs = []
		folder_back = "/home/kai/Desktop/Back/"
		files = listdir(folder_back, pattern = "*.png")
		folder_out = "/home/kai/Downloads/Sample/"

		for f in files: imgs.append(cv2.imread(folder_back + f))
		bgs = get_backgrounds(imgs)

		src = cv2.imread("/home/kai/Desktop/Work/img_2023_0213_154214_260.png")

		tmp = cv2.GaussianBlur(src, (5, 5), 3)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations = 5)
		cv2.imwrite(folder_out + "pre_process.jpeg", tmp)
		tmp_h = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)[:, :, 0]
		cv2.imwrite(folder_out + "pre_HSV_h.jpeg", tmp_h)

		tmp_g = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(folder_out + "pre_gray.jpeg", tmp_g)

		tmp = get_vote(src, bgs)
		cv2.imwrite(folder_out + "vote.jpeg", tmp)

		dst, edge, corners = test_scan(src, bgs)
		cv2.imwrite(folder_out + "detected_corners.jpeg", dst)
		cv2.imwrite(folder_out + "detected_edges.jpeg", edge)

		dst = warp(src, corners, 17.68, 25.70, 0, 330, [0, 1, 2, 3])
		cv2.imwrite(folder_out + "warped.jpeg", dst)

		# mask_list = analyze_edge(image)
		# for i, mask in enumerate(mask_list):
		# 	cv2.imwrite(folder_out + f"0{i:03d}.jpeg", mask)
		#
		# select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		# # select = [0, 1, 2, 3, 5, 6, 7, 8, 11]
		# pts_proj = preScan(image, select)
		# cv2.imwrite(folder_out + f"2000.jpeg", draw_cornerList(image, pts_proj))
		#
		# folder_back = folder + "source/back/"
		# folder_work = folder + "source/sample/"
		# ext_in = "png"
		# ext_out = "jpeg"

		pass


	test()
	pass
