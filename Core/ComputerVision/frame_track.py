import cv2
import numpy as np

from Core.ComputerVision import normalize_universal

MATCH_MIN = 16
RECONSTRUCTION_MIN = 6


def match_optFlow(gray_prev, gray_next, Us, winsize, flow_min):
	"""
	:param gray_prev: previous gray scala image
	:param gray_next: current gray scala image
	:param Us: keypoint_sample
	:param winsize:
	:param flow_min:
	:return:
		flag: if succeed
		U0: kps in frame prev
		U1: kps in frame next
		idxes: kps indices in kps sample
		flow:
	"""
	flow = cv2.calcOpticalFlowFarneback(
		gray_prev, gray_next, None, 0.5, 7, winsize, 3, 7, 1.5,
		cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
	)[Us[:, 1], Us[:, 0]]
	# pretend got matched points
	scales = np.linalg.norm(flow, axis = 1)
	valid = scales > flow_min
	if np.sum(valid) < MATCH_MIN:
		return False, None, None, None

	idxes = np.arange(len(Us))[valid]
	U0 = np.asarray(Us[valid])
	U1 = U0 + flow[valid]
	return True, U0, U1, idxes


def match_corner(gray_prev, gray_next, winsize, dis_min, U0 = None):
	"""
	:param gray_prev: previous gray scala image
	:param gray_next: current gray scala image
	:param winsize:
	:param dis_min:
	:param U0:
	:return:
		flag: if succeed
		U0: kps in frame prev
		U1: kps in frame next
		valid:
	"""
	if U0 is None:
		# Shi-Tomasi corner detection para
		U0 = cv2.goodFeaturesToTrack(gray_prev, 800, 0.3, dis_min, blockSize = 7)
	else: U0 = np.asarray(U0, np.float32)
	U1, status, _ = cv2.calcOpticalFlowPyrLK(
		gray_prev, gray_next, U0, None, maxLevel = 3,
		winSize = (winsize, winsize),
	)
	U0 = np.asarray(U0)
	U1 = np.asarray(U1)
	valid = status.ravel() > 0
	if np.sum(valid) < MATCH_MIN:
		return False, None, None, None
	return True, U0, U1, valid


# def feature_matcher(use_orb = True, use_affine = False):
# 	# cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
# 	# cv.FlannBasedMatcher(index_params, search_params)
# 	if use_orb == 'orb':
# 		match_conf = 0.3
# 	else:
# 		match_conf = 0.65
# 	if use_affine:
# 		matcher = cv2.detail_AffineBestOf2NearestMatcher(False, False, match_conf)
# 	else:
# 		matcher = cv2.detail.BestOf2NearestMatcher_create(False, match_conf)
# 	return matcher


# def feature_detector(name: str):
# 	feature_dict = {
# 		"sift": cv2.SIFT_create,
# 		"surf": cv2.xfeatures2d.SURF_create,
# 		"brisk": cv2.BRISK_create,
# 		"akaze": cv2.AKAZE_create,
# 		"orb": cv2.ORB_create,
# 	}
# 	name = name.lower()
# 	if name not in feature_dict.keys(): name = "orb"
# 	return feature_dict[name]()


# def match_feature(gray_prev, gray_next, detector, matcher, fea0: cv2.detail_ImageFeatures = None):
# 	"""
# 	todo: untested ..
# 	:param gray_prev:
# 	:param gray_next:
# 	:param detector:
# 	:param matcher:
# 	:param fea0:
# 	:return: flag, pts1, pts2 (normalized frame 2D point)
# 	"""
# 	if fea0 is None:
# 		fea0 = cv2.detail.computeImageFeatures(detector, [gray_prev])[0]
# 	fea1 = cv2.detail.computeImageFeatures(detector, [gray_next])[0]
#
# 	pairs = matcher.apply(fea0, fea1)
# 	# print(pairs.num_inliers)
# 	# if pairs.num_inliers < MATCH_MIN:
# 	# 	return False, None, None, None, None
#
# 	matches = pairs.getMatches()
# 	idxes0 = [m.queryIdx for m in matches]
# 	idxes1 = [m.trainIdx for m in matches]
# 	return True, fea0, fea1, idxes0, idxes1


# def sample_idxes(w, h, step: int = 10):
# 	"""
# 	:param w: image width
# 	:param h: image height
# 	:param step: sample width
# 	:return:
# 	"""
# 	assert step > 1
#
# 	from Core.Basic import errPrint
# 	errPrint("warning, abandoned function ..")
# 	start = step // 2
# 	w = w // step + 1
# 	h = h // step + 1
# 	y, x = np.indices((h, w))
# 	x = x.ravel()
# 	y = y.ravel()
# 	U = np.asarray([x, y], int).T
# 	return U * step + start


def __distribute_keypoint(U_src, idxes, Us, block_size, image_width):
	"""
	warming, this function is highly bound to CVCamera.frame_idxes(step)
	:param U_src: keypoint location in image
	:param idxes: sorted keypoint indices,
	:param Us: keypoint sample
	:param block_size: related to Us, be sure same to Us's step
	:param image_width:
	:return:
		U_dst: distribute keypoint
		valid:
		pair_ids:
			** usage **:
				U_target': same shape as Us
				U_target'[valid] = U_target[pair_ids]
	"""
	block_step = int(image_width / block_size)
	total = len(Us)
	valid = np.full(total, False)
	pair_ids = np.zeros(total, int)
	U_dst = np.copy(Us)
	if idxes is None:
		idxes = np.argsort(np.random.random(len(U_src)))
	# tmp_U = (U_src / block_size).astype(int)
	# Js = tmp_U.dot([[block_step], [1]]).astype(int)
	for i in idxes:
		# j = Js[i]
		j_clm = int(U_src[i, 0] // block_size)
		j_row = int(U_src[i, 1] // block_size)
		j = int(j_row * block_step + j_clm)
		if valid[j]: continue
		U_dst[j, :] = U_src[i]
		pair_ids[j] = i
		valid[j] = True
	return U_dst, valid, pair_ids[valid]


def distribute_keypoint(U_src, idxes, image_size, win_size):
	"""
	warming, this function is highly bound to CVCamera.frame_idxes(step)
	:param U_src: keypoint location in image
	:param idxes: sorted keypoint indices,
	:param image_size:
	:param win_size:
	:return:
		pair_ids:
	"""

	w = image_size[0] // win_size + 1
	h = image_size[1] // win_size + 1
	total = int(w * h)

	valid = np.full(total, False)
	pair_ids = np.zeros(total, int)
	U_tmp = (U_src // win_size).astype(int)
	Js = U_tmp[:, 0] * h + U_tmp[:, 1]

	if idxes is None:
		idxes = np.argsort(np.random.random(len(U_src)))
	for i in idxes:
		j = Js[i]
		if valid[j]: continue
		pair_ids[j] = i
		valid[j] = True
	return pair_ids[valid]


# def __recover_pose(U0, U1, CM, repeat = False):
# 	"""
# 	E = (C1 * K1)^t * F * (C2 * K2)
# 	:param U0:
# 	:param U1:
# 	:param CM:
# 	:param repeat:
# 	:return:
# 		flag:
# 		R: from 1 -> 0
# 		T: from 1 -> 0
# 		valid:
# 	"""
# 	if len(U0) < MATCH_MIN:
# 		return False, None, None, None
#
# 	U0_, K0 = normalize_keypoint(U0)
# 	U1_, K1 = normalize_keypoint(U1)
# 	flag, E, R, t, mask = cv2.recoverPose(
# 		U0_, U1_, K0.dot(CM), None, K1.dot(CM), None,
# 		method = cv2.RANSAC, threshold = 0.007, prob = 0.99,
# 	)
# 	valid = mask.ravel() > 0
# 	if not repeat or np.sum(valid) < MATCH_MIN:
# 		return flag, R, t.ravel(), valid
#
# 	U0_, K0 = normalize_keypoint(U0[valid])
# 	U1_, K1 = normalize_keypoint(U1[valid])
# 	flag_, E_, R_, t_, mask = cv2.recoverPose(
# 		U0_, U1_, K0.dot(CM), None, K1.dot(CM), None,
# 		method = cv2.RANSAC, threshold = 0.007 * 0.95, prob = 0.99,
# 	)
# 	if not flag_:
# 		return flag, R, t.ravel(), valid
# 	valid[valid] = mask.ravel() > 0
# 	return flag_, R_, t_.ravel(), valid


def recover_pose(U0, U1):
	"""
	tUi' = Ki * tUi
	tUi' = Ki * Ci * tP = Ci' tP
	E = (Ki * Ci)^-t * F * (Ki * Ci)^-1
	:param U0:
	:param U1:
	:return:
		flag:
		R: from 1 -> 0
		T: from 1 -> 0
		valid: inlier
	"""
	U0_, K0, inv_K0 = normalize_universal(U0)
	U1_, K1, inv_K1 = normalize_universal(U1)
	flag, E, R, t, mask = cv2.recoverPose(
		U1_, U0_, K1, None, K0, None, method = cv2.RANSAC,
		threshold = 0.007, prob = 0.99,
	)
	if not flag: return False, None, None, None
	return flag, R, t.ravel(), mask.ravel() > 0


def reconstruction(R, T, CM, U0, U1, *, z_near: float = 1, z_far: float = 40):
	"""
	reconstruction
		C * E * X - u0^ * s0 =  0
		C * R * X - u1^ * s1 = -C * T
	solve: AX = B
		| C    -u0^   0   | X^ = |  0  |
		| CR    0    -u1^ |      | -CT |
		X^ = (x, y, z, s0, s1)
	:param R:
	:param T:
	:param CM:
	:param U0:
	:param U1:
	:param z_near:
	:param z_far:
	:return:
		X_assume:
		idxes:
	"""
	CR = CM.dot(R)
	CT = T.dot(CM.T)

	A = np.zeros((6, 5))
	A[0:3, :3] = CM
	A[3:6, :3] = CR
	A[2, 3] = -1
	A[5, 4] = -1
	B = np.zeros(6)
	B[3:6] = -CT
	idxes = []
	X_assume = []
	for i in range(len(U0)):
		A[0, 3] = -U0[i, 0]
		A[1, 3] = -U0[i, 1]
		A[3, 4] = -U1[i, 0]
		A[4, 4] = -U1[i, 1]
		inv_AA = np.linalg.inv(A.T.dot(A))
		x, y, z, s0, s1 = B.dot(A).dot(inv_AA.T)
		if s0 < z_near or s0 > z_far: continue
		if s1 < z_near or s1 > z_far: continue
		idxes.append(i)
		X_assume.append([x, y, z])
	length = len(idxes)
	if length < 1: return None, None
	X_assume = np.asarray(X_assume)
	idxes = np.asarray(idxes)
	return X_assume, idxes


def reconstruction_3f(R10, T10, R20, T20, CM, U0, U1, U2, *, z_near: float = 1, z_far: float = 40,
                      k_ratio: float = 0.5):
	"""
	reconstruction
		C *  E  * X - u0^ * s0                  =  0
		C * R10 * X - u1^ * s1                  = -C * T10
		C * R20 * X - u1^ * s2 + C * T20 * k20  =  0
	solve: AX = B, dof = -2
		| C     -u0^   0     0     0   | X^ = |   0   |
		| CR10   0    -u1^   0     0   |      | -CT10 |
		| CR20   0     0    -u2^  CT20 |      |   0   |
		X^ = (x, y, z, s0, s1, s2, k20)
	:param R10:
	:param T10:
	:param R20:
	:param T20:
	:param CM:
	:param U0:
	:param U1:
	:param U2:
	:param z_near:
	:param z_far:
	:param k_ratio:
	:return:
		X_assume:
		idxes:
		k20:
	"""
	CR10 = CM.dot(R10)
	CR20 = CM.dot(R20)
	CT10 = T10.dot(CM.T)
	CT20 = T20.dot(CM.T)

	A = np.zeros((9, 7))
	A[0:3, :3] = CM
	A[3:6, :3] = CR10
	A[6:9, :3] = CR20
	A[6:9, -1] = CT20
	A[2, 3] = -1
	A[5, 4] = -1
	A[8, 5] = -1
	B = np.zeros(9)
	B[3:6] = -CT10
	idxes = []
	X_assume = []
	k20 = 0
	for i in range(len(U0)):
		A[0, 3] = -U0[i, 0]
		A[1, 3] = -U0[i, 1]
		A[3, 4] = -U1[i, 0]
		A[4, 4] = -U1[i, 1]
		A[6, 5] = -U2[i, 0]
		A[7, 5] = -U2[i, 1]
		inv_AA = np.linalg.inv(A.T.dot(A))
		x, y, z, s0, s1, s2, k = B.dot(A).dot(inv_AA.T)
		if s0 < z_near or s0 > z_far: continue
		if s1 < z_near or s1 > z_far: continue
		if s2 < z_near or s2 > z_far: continue
		if abs(k - 1) > k_ratio: continue
		idxes.append(i)
		X_assume.append([x, y, z])
		k20 += k
	length = len(idxes)
	if length < MATCH_MIN: return None, idxes, 1
	X_assume = np.asarray(X_assume)
	k20 /= length
	return X_assume, idxes, k20


def decompose_Essential(E, U0, U1, CM):
	"""
	todo: func for reference only, and even not tested ..
	:param E: Essential Matrix
	:param U0: kps in frame prev
	:param U1: kps in frame next
	:param CM: camera matrix
	:return:
		flag: if succeed
		R:
		T:
		valid: good kps valid
	"""
	u, s, vh = np.linalg.svd(E)
	t = u[:, 2]
	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
	R1 = u.dot(W).dot(vh)
	R2 = u.dot(W.T).dot(vh)
	if np.linalg.det(R1) < 0:
		R1 = -R1
		R2 = -R2

	candies = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
	valids = np.full((4, len(U0)), False)
	counts = []
	for i, (R, T) in enumerate(candies):
		X, idxes = reconstruction(R, T, CM, U0, U1)
		counts.append(len(idxes))
		valids[i, idxes] = True
	best_case = np.argmax(counts)
	if counts[best_case] < 16:
		return False, None, None, None
	R, T = candies[best_case]
	valid = valids[best_case]
	return True, R, T, valid

# if __name__ == '__main__':
# 	from Core.ComputerVision import CVCamera
# 	from Core.Visualization import KaisCanvas, SpaceCanvas, KaisColor
# 	from Core.Geometry import CoordSys
#
# 	sCanvas = SpaceCanvas("/home/kai/PycharmProjects/pyCenter/d_2022_0822/out")
# 	canvas2 = KaisCanvas()
#
#
# 	def get_images():
# 		path = "/home/kai/PycharmProjects/pyCenter/d_2022_0816/source/210302013247/20210302013546365000.jpg"
# 		img_prev = cv2.imread(path)
# 		path = "/home/kai/PycharmProjects/pyCenter/d_2022_0816/source/210302013247/20210302013546568000.jpg"
# 		img_curr = cv2.imread(path)
# 		path = "/home/kai/PycharmProjects/pyCenter/d_2022_0816/source/210302013247/20210302013546772000.jpg"
# 		img_next = cv2.imread(path)
# 		return img_prev, img_curr, img_next
#
#
# 	def get_info():
# 		image_size = 1280, 720
# 		CB = CVCamera(616.5263, 617.2315, 651.1589, 376.0408, image_size)
# 		return image_size, CB
#
#
# 	def test_optFlow():
# 		image_size, CB = get_info()
#
# 		Us = CB.frame_idxes(20)
# 		winsize = 50
# 		flow_min = 20
#
# 		img_prev, img_curr, img_next = get_images()
# 		gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
# 		gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
# 		gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
# 		# ========= ========= ========= ========= =========
# 		flag, U0a, U1, idxes1 = match_optFlow(gray_curr, gray_prev, Us, winsize, flow_min)
# 		assert flag
# 		flag, R10, T10, valid1 = recover_pose(U0a, U1, CB.M)
# 		assert flag
# 		print(f"left about {np.sum(valid1) / len(valid1):.3f}")
# 		idxes1 = idxes1[valid1]
# 		U0a = U0a[valid1]
# 		U1 = U1[valid1]
# 		# ========= ========= ========= ========= =========
# 		flag, U0b, U2, idxes2 = match_optFlow(gray_curr, gray_next, Us, winsize, flow_min)
# 		assert flag
# 		flag, R20, T20, valid2 = recover_pose(U0b, U2, CB.M)
# 		assert flag
# 		print(f"left about {np.sum(valid2) / len(valid2):.3f}")
# 		idxes2 = idxes2[valid2]
# 		U0b = U0b[valid2]
# 		U2 = U2[valid2]
# 		# ========= ========= ========= ========= =========
#
# 		canvas2.ax.imshow(gray_curr[:, :, ::-1], alpha = 0.5)
# 		canvas2.draw_line(U0a, U1, color = "pink", alpha = 0.5)
# 		canvas2.draw_line(U0b, U2, color = "pink", alpha = 0.5)
#
# 		idxes, valid1, valid2 = np.intersect1d(idxes1, idxes2, True, True)
# 		U0 = U0b[valid2]
# 		U1 = U1[valid1]
# 		U2 = U2[valid2]
#
# 		canvas2.draw_line(U0, U1, color = "orange")
# 		canvas2.draw_line(U0, U2, color = "skyblue")
#
# 		canvas2.set_axis(xlim = (-100, 1380), grid_on = False)
# 		canvas2.show()
#
# 		sCanvas.add_wireCamera("C0", s = 0.5)
# 		R01, T01 = CoordSys.inv(R10, T10)
# 		sCanvas.add_wireCamera("C1", s = 0.5, R = R01, T = T01)
# 		R02, T02 = CoordSys.inv(R20, T20)
# 		sCanvas.add_wireCamera("C2", s = 0.5, R = R02, T = T02)
#
# 		X_assume, idxes, k20 = reconstruction_3f(R10, T10, R20, T20, CB.M, U0, U1, U2)
# 		print(f"left about {len(idxes)} / {len(U0)} = {len(idxes) / len(U0):.3f}")
# 		print(k20)
#
# 		R02, T02 = CoordSys.inv(R20, T20)
# 		sCanvas.add_wireCamera("C2x", s = 0.5, R = R02, T = T02 * k20)
#
# 		sCanvas.add_point("Pts", X_assume, color = KaisColor.plotColor("white"))
# 		X_assume1, idxes = reconstruction(R10, T10, CB.M, U0, U1)
# 		sCanvas.add_point("Pts_", X_assume1, color = KaisColor.plotColor("blue"))
# 		X_assume2, idxes = reconstruction(R20, T20 * k20, CB.M, U0, U2)
# 		sCanvas.add_point("Pts_", X_assume2, color = KaisColor.plotColor("green"))
#
#
# 	def learn_corner():
# 		image_size, CB = get_info()
#
# 		img_prev, img_curr, img_next = get_images()
# 		# gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
# 		gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
# 		# gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
#
# 		# ========= ========= ========= ========= >> Harris Corner
# 		dst = cv2.cornerHarris(gray_curr, 2, 3, 0.04)
# 		# Threshold for an optimal value, it may vary depending on the image.
# 		# img_curr[dst > 0.01 * dst.max()] = KaisColor.plotColor("crimson", True)
# 		ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
# 		# find centroids and refine the corners
# 		ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
# 		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# 		U_h = cv2.cornerSubPix(gray_curr, np.float32(centroids), (5, 5), (-1, -1), criteria)
# 		# ========= ========= ========= ========= >> Shi-Tomasi Corner
# 		U_st = cv2.goodFeaturesToTrack(gray_curr, 0, 0.01, 10)
# 		U_st = U_st.reshape((-1, 2))
#
# 		# ========= ========= ========= =========
# 		canvas2.ax.imshow(img_curr[:, :, ::-1])
# 		# canvas2.draw_points(U_h, color = "red", marker = ".", s = 10)
# 		canvas2.draw_points(U_st, color = "pink", marker = ".", s = 10)
#
# 		canvas2.set_axis(xlim = (-30, 1310), grid_on = False)
# 		# canvas2.set_axis(equal_axis = False)
# 		canvas2.show()
# 		canvas2.clear()
# 		pass
#
#
# 	def learn_feature():
# 		image_size, CB = get_info()
#
# 		img_prev, img_curr, img_next = get_images()
# 		gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
# 		gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
# 		# gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
#
# 		# ========= ========= ========= ========= >> detector
# 		detector = feature_detector("SURF")
#
# 		# ========= ========= ========= ========= >> get features
# 		fea0 = cv2.detail.computeImageFeatures(detector, [gray_curr])[0]
# 		kpt0 = fea0.keypoints
# 		des0 = fea0.descriptors
# 		fea1 = cv2.detail.computeImageFeatures(detector, [gray_prev])[0]
# 		kpt1 = fea1.keypoints
# 		des1 = fea1.descriptors
#
# 		# ========= ========= ========= ========= >> matching
# 		# --------- BFMatcher ---------
# 		# --------- plan 1: normal match (ORB)
# 		# bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
# 		# matches = bfm.match(des0, des1)
# 		# matches = sorted(matches, key = lambda x: x.distance)
# 		# U0 = [kpt0[m.queryIdx].pt for m in matches]
# 		# U1 = [kpt1[m.trainIdx].pt for m in matches]
# 		# --------- plan 2: knn match (SIFT)
# 		# bfm = cv2.BFMatcher()
# 		# matches = bfm.knnMatch(des0, des1, k = 2)
# 		# ratio test as per Lowe's paper
# 		# good_matches = []
# 		# for m, n in matches:
# 		# 	if m.distance < 0.75 * n.distance:
# 		# 		good_matches.append(m)
# 		# U0 = [kpt0[m.queryIdx].pt for m in good_matches]
# 		# U1 = [kpt1[m.trainIdx].pt for m in good_matches]
# 		# --------- FLANN Matcher ---------
# 		# >> ORB
# 		# FLANN_INDEX_LSH = 6
# 		# index_params = dict(
# 		# 	algorithm = FLANN_INDEX_LSH,
# 		# 	table_number = 6,  # 12
# 		# 	key_size = 12,  # 20
# 		# 	multi_probe_level = 1,  # 2
# 		# )
# 		# >> SIFT, SURF etc.
# 		FLANN_INDEX_KDTREE = 1
# 		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# 		# --------- ---------
# 		search_params = dict(checks = 50)  # or pass empty dictionary
# 		flann = cv2.FlannBasedMatcher(index_params, search_params)
# 		matches = flann.knnMatch(des0, des1, k = 2)
# 		# ratio test as per Lowe's paper
# 		good_matches = []
# 		for m, n in matches:
# 			if m.distance < 0.75 * n.distance:
# 				good_matches.append(m)
# 		U0 = [kpt0[m.queryIdx].pt for m in good_matches]
# 		U1 = [kpt1[m.trainIdx].pt for m in good_matches]
# 		print(U0)
#
# 		# ========= ========= ========= ========= =========
# 		canvas2.ax.imshow(img_curr[:, :, ::-1], alpha = 0.5)
# 		canvas2.draw_line(U0, U1, color = "orange", lw = 1.0)
# 		# canvas2.draw_points([ki.pt for ki in kpt0], marker = ".", color = "yellow", s = 10)
#
# 		canvas2.set_axis(xlim = (-30, 1310), grid_on = False)
# 		canvas2.show()
# 		canvas2.clear()
#
# 		pass
#
#
# 	def main():
# 		# test_optFlow()
# 		learn_feature()
# 		# learn_corner()
# 		pass
#
#
# 	main()
# 	sCanvas.save()
# 	pass
