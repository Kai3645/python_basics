import math

import cv2
import numpy as np
from scipy import stats

# from src.basic_funcs import perspective, printLog
# from support import Config
#
# LANE_LINE_WIDTH = Config.lane_line_width_real
# LANE_WIDTH_BASIC = Config.lane_width_basic_real
# PIXEL_COE = Config.laneDet_pixel_coe
# SCAN_LENGTH = Config.laneDet_scan_length
# LANE_GAP_LEN = Config.laneDet_lane_gap_length
# LANE_MIN_LEN = Config.laneDet_lane_min_length
# LANE_BASIC_LEN = Config.laneDet_lane_basic_length
# LANE_IMAGE_LEN_TH = Config.laneDet_lane_image_min_length
#
# LANE_DETECT_TH = PIXEL_COE ** 2 * LANE_LINE_WIDTH * LANE_BASIC_LEN * 0.5  # with 50% visible
# LANE_HIST_ITERATIONS = max(1, int(round(PIXEL_COE / 10)))
#
# LINE_W = int(LANE_LINE_WIDTH * PIXEL_COE)
# LANE_W = int(LANE_WIDTH_BASIC * PIXEL_COE)
# WIN_BASIC_W = LINE_W
# WIN_BASIC_H = int(PIXEL_COE * SCAN_LENGTH)
# WIN_STP_SIZE = int(LINE_W / 2)
# WIN_WIDTH_TH = LANE_W / 4
# WIN_GAP_CD = int(LANE_GAP_LEN / SCAN_LENGTH)
# WIN_LEN_CD = int(LANE_MIN_LEN / SCAN_LENGTH)
#
# printLog("LaneDet", f"gap cd = {WIN_GAP_CD}, len cd = {WIN_LEN_CD}")


class ScanWindow:
	def __init__(self, box_size):
		"""
		:param box_size: (width, height)
		:return:
		"""
		# once init, will not change
		self._box_w = box_size[0]
		self._box_h = box_size[1]
		# identity paras, using left top corner for locating
		self.x = 0
		self.y = 0
		self.w = WIN_BASIC_W  # as default
		self.h = WIN_BASIC_H  # unchangeable

	def fit_box(self):
		"""
		cut out the part, where is outside
		:return: self
		"""
		# cut left
		if self.x < 0:
			self.w += self.x
			self.x = 0
		# cut right
		if self.x + self.w > self._box_w:
			self.w = self._box_w - self.x
		# not cut top, move down
		if self.y < 0: self.y = 0
		# not cut bottom, move up
		if self.y + self.h > self._box_h:
			self.y = self._box_h - self.h
		return self

	def setup(self, cx: int, cy: int, w: int):
		"""
		:param cx: center top x
		:param cy: center top y
		:param w: width
		"""
		self.x = cx - w // 2
		self.y = cy
		self.w = w
		return self.fit_box()

	def __str__(self):
		return f"({self.x}, {self.y}, {self.w}, {self.h})"

	def pt1(self):
		return self.x, self.y

	def pt2(self):
		return self.x + self.w, self.y + self.h

	def center_x(self):
		return self.x + self.w // 2

	def expand_width(self, hist):
		# expand left
		while self.x > 0:
			x = self.x
			if hist[x] < 1: break
			self.x -= WIN_STP_SIZE
			self.w += WIN_STP_SIZE
		# expand right
		while self.x < self._box_w - self.w:
			x = self.x + self.w
			if hist[x] < 1: break
			self.w += WIN_STP_SIZE
		# fit box and check width
		self.fit_box()
		if self.w > WIN_WIDTH_TH: return False
		return True

	def move_to(self, cx: int, cy: int, binary):
		"""
		:param cx: center top x
		:param cy: center top y, warming: 'cy > box height' will not check
		:param binary:
		:return:
			flag: if find good cx
			cx: new center of the window
		"""
		cx_old = cx

		# move to cy
		self.setup(cx, cy, LANE_W // 4)

		y1 = self.y = cy
		y2 = cy + self.h
		# scan by x-axis
		hist = np.sum(binary[y1:y2], axis = 0) / 255
		hist = np.convolve(hist, (1, 1, 1), "same")

		# find new center x by old window range
		sub_hist = hist[self.x:self.x + self.w]
		pixCount = sum(sub_hist)
		if pixCount < 10:
			return False, cx_old  # empty window
		local_cx = np.argmax(sub_hist)
		if sub_hist[local_cx] < 10:
			return False, cx_old  # empty window
		cx_new = self.x + local_cx
		if abs(cx_new - cx_old) > WIN_BASIC_W * 2:
			return False, cx_old  # over moved
		self.setup(cx_new, self.y, WIN_BASIC_W)
		if not self.expand_width(hist):
			return False, cx_old  # bad window

		# de-noisy by pca
		x1, y1 = self.pt1()
		x2, y2 = self.pt2()
		ys, xs = np.nonzero(binary[y1:y2, x1:x2])
		# from Core.Statistic import PCA
		# _, _, u = PCA(np.asarray([xs, ys]).T)  # slow but controllable for debug
		u, _, _ = np.linalg.svd(np.cov([xs, ys]))  # fast but will pass some errors
		a, b = np.abs(u[:, -1])  # todo: find out which is main direction, i just forgot
		if b > a * 0.25:  # about 15 [deg]
			return False, cx_old
		return True, self.center_x()


class LaneCache:
	def __init__(self):
		self._gap_cd = WIN_GAP_CD
		self._len_cd = WIN_LEN_CD

		self.rects = []  # for saving (x1, y1, x2, y2)
		self.centers = []  # for saving (cx, cy)

	def add(self, rect, center):
		self.rects.append(rect)
		self.centers.append(center)
		self._len_cd -= 1
		self._gap_cd = WIN_GAP_CD

	def add_gap(self):
		self._gap_cd -= 1

	def need_divide(self):
		"""
		:return: flag: if lane need divide
		"""
		if self._gap_cd > 0: return False  # go on
		if self._len_cd <= 0: return True  # long enough
		# bad case, too mach empty space and not long enough
		# setup self and go on as a new object
		self._gap_cd = WIN_GAP_CD
		self._len_cd = WIN_LEN_CD
		self.rects = []
		self.centers = []
		return False

	def is_good(self):
		"""
		:return: if lane is long enough
		"""
		return self._len_cd <= 0

	def push(self, binary):
		mask = np.zeros_like(binary)
		for rect in self.rects:
			x1, y1, x2, y2 = rect
			mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
		return mask, self.centers

	def centroid(self, binary):
		pts = []
		mask = np.zeros_like(binary)
		for rect in self.rects:
			x1, y1, x2, y2 = rect
			ys, xs = np.nonzero(binary[y1:y2, x1:x2])
			mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
			pts.append([np.average(xs) + x1, np.average(ys) + y1])
		return mask, np.asarray(pts)


class LaneDetector:
	"""
	purpose: given image return lane points, un-straight line or over detect would be fine

	Reference:
	[1] 'awesome-lane-detection',
		https://github.com/amusi/awesome-lane-detection
	[2] 'Lane Line Detection using Python and OpenCV',
		https://github.com/tatsuyah/Lane-Lines-Detection-Python-OpenCV

	memo:
	1) for two kind of perspective view
		- original(camera) view:
			- '_i' for points/matrix
			- 'orig_' for image
		- ground(vehicle) view:
			- '_g' for points/matrix
			- 'pers_' for image
	"""

	def __init__(self, persMat_i2g, size_i, size_g):
		"""
		:param persMat_i2g: perspective matrix i -> g
		:param size_i: image size before perspective, i -> image
		:param size_g: image size before perspective, g -> ground
		"""
		self.persMat_i2g = persMat_i2g
		self.persMat_g2i = np.linalg.inv(persMat_i2g)
		self.size_i = size_i
		self.size_g = size_g

		# storage para
		self.ground_intensity_th = -1

		self._orig_image = None
		self.sum_yellow = np.zeros(3)
		self.num_yellow = 0
		self.sum_white = np.zeros(3)
		self.num_white = 0

		# self._origValid_yellow = None
		# self._origValid_white = None
		# self.yellow_value = None
		# self.white_value = None

		# ---------- debug ----------
		self.pers_lane = None
		self.orig_lane = None
		# ---------------------------
		pass

	def warp_i2g(self, image):
		return cv2.warpPerspective(image, self.persMat_i2g, self.size_g, borderMode = cv2.BORDER_REPLICATE)

	def warp_g2i(self, image):
		return cv2.warpPerspective(image, self.persMat_g2i, self.size_i)

	def get_persMask_lane(self):
		"""
		short names: todo: add statement
			- 'mask'/'binary'
			- 'valid'/'invalid'
			- 'gray'/'YCrCb'/'HLS'
			- 'grad'
		:return:
			_: mask of possible pixel, where lane could be, in ground view
		"""
		# ========== setup image ==========
		orig_image = np.copy(self._orig_image)  # for safety
		orig_gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
		# get ground range in origin image
		persMask_ground = np.full((self.size_g[1], self.size_g[0]), 255, np.uint8)
		origMask_ground = self.warp_g2i(persMask_ground)
		valid_ground = origMask_ground > 200  # not necessary num
		invalid_ground = np.logical_not(valid_ground)

		# ========== color mask ==========
		# 1) gray mask
		gray_th = stats.mode(orig_gray[valid_ground], axis = None)[0][0]
		gray_th = gray_th * 15 / 16 + 16  # todo: delete if did not work for night images
		if self.ground_intensity_th > 0:  # let ground intensity threshold stable
			gray_th = self.ground_intensity_th * 0.34 + gray_th * 0.66
		self.ground_intensity_th = gray_th
		orig_gray_fix = np.copy(orig_gray)
		orig_gray_fix[orig_gray < gray_th] = gray_th  # remove shadow
		orig_gray_fix[invalid_ground] = gray_th  # remove out of view
		_, origMask_gray = cv2.threshold(orig_gray_fix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		# 2) YCrCb mask, almost for yellow/orange detect
		orig_YCrCb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2YCrCb)
		orig_YCrCb_Cr = cv2.equalizeHist(orig_YCrCb[:, :, 1])
		origMask_Cr = np.zeros((self.size_i[1], self.size_i[0]), np.uint8)
		origMask_Cr[orig_YCrCb_Cr > 250] = 255
		orig_YCrCb_Cb = cv2.equalizeHist(orig_YCrCb[:, :, 2])
		origMask_Cb = np.zeros((self.size_i[1], self.size_i[0]), np.uint8)
		origMask_Cb[orig_YCrCb_Cb < 5] = 255
		origMask_YCrCb = np.bitwise_and(origMask_Cr, origMask_Cb)
		origMask_YCrCb[invalid_ground] = 0

		# 3) HLS_S mask, almost for yellow/orange detect
		orig_HLS = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HLS)
		origMask_S = np.zeros((self.size_i[1], self.size_i[0]), np.uint8)
		origMask_S[orig_HLS[:, :, 2] > 180] = 255  # todo: some time not works well
		origMask_S[invalid_ground] = 0

		# 4) combine color masks then project to ground view
		origMask_color = np.bitwise_or(origMask_gray, origMask_YCrCb)
		origMask_color = np.bitwise_or(origMask_color, origMask_S)
		pers_mask_color = self.warp_i2g(origMask_color)
		persMask_color = np.zeros((self.size_g[1], self.size_g[0]), np.uint8)
		persMask_color[pers_mask_color > 128] = 255

		# ========== gradient mask ==========
		def sobel_thresh(gary):  # for ground view image only
			sobel = np.abs(cv2.Sobel(gary, cv2.CV_32F, 1, 0, ksize = 5))  # x(w)-axis gradient
			scale = sobel / np.max(sobel)
			mask = np.zeros((self.size_g[1], self.size_g[0]), np.uint8)
			mask[(scale > 0.15) & (scale < 0.85)] = 255
			mask = cv2.dilate(mask, None, iterations = 2)
			mask = cv2.erode(mask, None, iterations = 1)
			return mask

		# 1) gray gradient
		pers_gary = self.warp_i2g(orig_gray)  # do not use gray_fix
		grad_gray = sobel_thresh(pers_gary)

		# 2) HLS_S gradient
		pers_HLS_S = self.warp_i2g(orig_HLS[:, :, 2])
		grad_HLS_S = sobel_thresh(pers_HLS_S)

		# 3) combine all persMasks
		persMask_grad = np.bitwise_or(grad_gray, grad_HLS_S)
		persMask_combine = np.bitwise_and(persMask_color, persMask_grad)
		# ---------- local debug ----------
		# orig_
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_image.jpg", orig_image)
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_gray.jpg", orig_gray)
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_gray_fix.jpg", orig_gray_fix)
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_YCrCb_Cr.jpg", orig_YCrCb_Cr)
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_YCrCb_Cb.jpg", orig_YCrCb_Cb)
		# cv2.imwrite(folder_LaneDetect + "tmp/orig_HLS_S.jpg", orig_HLS[:, :, 2])
		# # origMask_
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_gray.jpg", origMask_gray)
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_Cr.jpg", origMask_Cr)
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_Cb.jpg", origMask_Cb)
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_YCrCb.jpg", origMask_YCrCb)
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_S.jpg", origMask_S)
		# cv2.imwrite(folder_LaneDetect + "tmp/origMask_color.jpg", origMask_color)
		# # pers_
		# cv2.imwrite(folder_LaneDetect + "tmp/pers_gary.jpg", pers_gary)
		# cv2.imwrite(folder_LaneDetect + "tmp/pers_HLS_S.jpg", pers_HLS_S)
		# cv2.imwrite(folder_LaneDetect + "tmp/pers_mask_color.jpg", pers_mask_color)
		# # persGrad_
		# cv2.imwrite(folder_LaneDetect + "tmp/persGrad_gray.jpg", grad_gray)
		# cv2.imwrite(folder_LaneDetect + "tmp/persGrad_HLS_S.jpg", grad_HLS_S)
		# # persMask_
		# cv2.imwrite(folder_LaneDetect + "tmp/persMask_color.jpg", persMask_color)
		# cv2.imwrite(folder_LaneDetect + "tmp/persMask_grad.jpg", persMask_grad)
		# cv2.imwrite(folder_LaneDetect + "tmp/persMask_combine.jpg", persMask_combine)
		# ---------------------------------
		return persMask_combine

	@staticmethod
	def get_lane_xCandies(binary):
		# calculate bottom half ground binary then convolve by lane width ones
		hist = np.sum(binary, axis = 0) / 255
		hist = np.convolve(hist, np.ones(LINE_W), mode = "same")
		dHist = np.convolve(hist, (1, 0, -1), mode = "same")
		# ---------- local debug ----------
		# from Core.Visualization import KaisCanvas
		# canvas = KaisCanvas(linewidth = 1.25)
		# canvas.ax.plot(hist)
		# canvas.ax.plot(dHist)
		# ---------------------------------
		dHist[(hist < LANE_DETECT_TH) | (dHist < 0)] = 0
		# initial paras
		candies = []
		# local_min = init_min = np.max(dHist)
		for i, (dh1, dh2) in enumerate(zip(dHist[:-1], dHist[1:])):
			if dh1 > dh2 <= 0: candies.append(i)
		# ---------- local debug ----------
		# for candy in candies:
		# 	canvas.draw_lines_p2p((candy, 0), (candy, hist[candy]),
		# 	                      para = dict(dashes = [8, 5, 8, 5], color = "crimson"))
		# 	canvas.draw_lines_p2p((0, LANE_DETECT_TH), (binary.shape[1], LANE_DETECT_TH),
		# 	                      para = dict(dashes = [12, 4, 5, 4], color = "white", lw = 1))
		# canvas.set_axis(equal_axis = False, sci_on = False)
		# canvas.save(folder_LaneDetect + "tmp/histogram.jpg")
		# canvas.close()
		# ---------------------------------
		return candies

	def color_meter(self, mask):
		"""
		:param mask:
		:return: flag: if is white/yellow
		"""
		ys, xs = np.nonzero(self.warp_g2i(mask))
		length_half = len(xs) // 2
		colors = self._orig_image[ys, xs]
		idxes = np.argsort(colors[:, 2])[length_half:]
		colors = colors[idxes]
		ave = np.mean(colors, axis = 0)

		if ave[2] < self.ground_intensity_th:  # bad color
			self.orig_lane[ys, xs] = (200, 50, 10)
			return False
		if max(ave) - min(ave) < 30:  # assume white
			self.sum_white += np.sum(colors, axis = 0)
			self.num_white += length_half
			self.orig_lane[ys, xs] = (255, 255, 255)
			return True
		if 128 > ave[2] - ave[1] > 0 and ave[0] < 128:  # maybe yellow
			self.sum_yellow += np.sum(colors, axis = 0)
			self.num_yellow += length_half
			self.orig_lane[ys, xs] = (0, 140, 255)
			return True
		# unknown color
		self.orig_lane[ys, xs] = (200, 50, 10)
		return False

	def recognize_lanes(self, binary, xCandies):
		# ---------- local debug ----------
		from Core.Visualization import KaisColor
		c_inlier = KaisColor.plotColor("limegreen", gbr = True).astype(int).tolist()
		c_outlier = KaisColor.plotColor("crimson", gbr = True).astype(int).tolist()
		# ---------------------------------

		STP_NUM = int(math.ceil(self.size_g[1] / WIN_BASIC_H))
		win = ScanWindow(self.size_g)

		laneCaches = []
		# scan for each lane candy
		for xCandy in xCandies:
			cx, cy = xCandy, self.size_g[1] - win.h
			win.setup(cx, cy, LANE_W // 4)
			laneCache = LaneCache()
			for _ in range(STP_NUM):
				flag, cx = win.move_to(cx, cy, binary)
				cy = win.y - win.h  # new cy
				if not flag:
					# ---------- local debug ----------
					cv2.rectangle(self.pers_lane, win.pt1(), win.pt2(), c_outlier, 2)
					# ---------------------------------
					laneCache.add_gap()  # empty/bad window
				else:
					x1, y1 = win.pt1()
					x2, y2 = win.pt2()
					# ---------- local debug ----------
					cv2.rectangle(self.pers_lane, (x1, y1), (x2, y2), c_inlier, 2)
					# ---------------------------------
					laneCache.add((x1, y1, x2, y2), (cx, (y1 + y2) / 2))
				if not laneCache.need_divide(): continue
				# save old cache and renew one
				laneCaches.append(laneCache)
				laneCache = LaneCache()
			# for last lane in loop
			if laneCache.is_good():
				laneCaches.append(laneCache)

		# combine results for case 1, pts by detect mask
		persMask_lane = np.zeros((self.size_g[1], self.size_g[0]), np.uint8)
		# sample_lanes_pts_i = []
		for lineCash in laneCaches:
			mask, pts_g = lineCash.push(binary)
			pts_i = perspective(self.persMat_g2i, pts_g)
			dis_i = np.linalg.norm(pts_i[0] - pts_i[-1])
			if dis_i < LANE_IMAGE_LEN_TH: continue
			if not self.color_meter(mask): continue
			persMask_lane[mask > 0] = 255
		# sample_lanes_pts_i.append(pts_i)

		# combine results for case 3, mask centroid of each scan window
		# lane_pts_g = np.zeros((0, 2))
		# for lineCash in laneCaches:
		# 	mask, pts_g = lineCash.centroid(binary)
		# 	pts_i = perspective(self.persMat_g2i, pts_g)
		# 	dis_i = np.linalg.norm(pts_i[0] - pts_i[-1])
		# 	if dis_i < LANE_IMAGE_LEN_TH: continue
		# 	if not self.color_meter(mask): continue
		# 	lane_pts_g = np.concatenate((lane_pts_g, pts_g))
		# return lane_pts_g
		return persMask_lane

	def run(self, img_org):  # PS. candy <=> candidate
		self._orig_image = np.copy(img_org)
		# origin image -> perspective mask where lane could be
		persMask_laneCandies = self.get_persMask_lane()

		# ---------- debug ----------
		self.orig_lane = np.copy(img_org)
		self.pers_lane = cv2.cvtColor(persMask_laneCandies, cv2.COLOR_GRAY2BGR) // 2
		# ---------------------------

		# clean up the perspective mask
		hist_binary = np.copy(persMask_laneCandies)  # for safety
		hist_binary = cv2.erode(hist_binary, None, iterations = 1)
		hist_binary = cv2.dilate(hist_binary, None, iterations = LANE_HIST_ITERATIONS)
		hist_binary = cv2.erode(hist_binary, None, iterations = LANE_HIST_ITERATIONS - 1)
		# if DEBUG: cv2.imwrite(folder_LaneDetect + "tmp/histogram_input.jpg", hist_binary)

		# -> x(w)-axis coordinate where lane could be
		xCandies_1 = self.get_lane_xCandies(hist_binary[int(self.size_g[1] * 0.4):])
		xCandies_2 = self.get_lane_xCandies(hist_binary[int(self.size_g[1] * 0.7):])
		xCandies_tmp = xCandies_1 + xCandies_2
		if len(xCandies_tmp) == 0: return None
		xCandies_tmp = np.sort(xCandies_tmp)
		xCandies = [xCandies_tmp[0]]
		for candy in xCandies_tmp:
			if candy - xCandies[-1] < WIN_BASIC_W: continue
			xCandies.append(candy)
		printLog("LaneDet", f"xCandies = {xCandies}")

		# todo: 3 kind of lane pts, figure out which is better
		#   1, pts by detect mask << current
		#   2, centers of scan window
		#   3, mask centroid of each scan window

		persMask_lane = self.recognize_lanes(persMask_laneCandies, xCandies)
		org_mask_lane = self.warp_g2i(persMask_lane)
		origValid_lane = org_mask_lane > 128
		if np.sum(origValid_lane) == 0: return None
		ys, xs = np.nonzero(origValid_lane)
		lane_pts_i = np.asarray([xs, ys]).T

		# lane_pts_g = self.recognize_lanes(persMask_laneCandies, xCandies)
		# if len(lane_pts_g) == 0: return None
		# lane_pts_i = perspective(self.persMat_g2i, lane_pts_g)

		# ---------- local debug ----------
		# cv2.imwrite(folder_LaneDetect + "tmp/pers_lane.jpg", persMask_lane)
		# cv2.imwrite(folder_LaneDetect + "tmp/org_lane.jpg", self.orig_lane)
		# ---------------------------------
		return lane_pts_i

	pass
