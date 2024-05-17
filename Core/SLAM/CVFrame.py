import os

import cv2
import numpy as np
import yaml

from Core.Basic import KaisLog
from Core.ComputerVision.Replaceable import remapping, project_flat, recover_flat, project_cylinder, recover_cylinder
from Core.ComputerVision.frame_track import distribute_keypoint

log = KaisLog.get_log()

MATCH_MIN = 10
MATCH_GOOD = 200
DISTANCE_MIN = 10
DISTANCE_MIN2 = DISTANCE_MIN * DISTANCE_MIN


class CVFrame:
	@staticmethod
	def get_cvf_config(path: str = None):
		if path is None:
			loc = os.path.dirname(__file__)
			path = loc + "/default_config/cvf_kwargs.yaml"
		config = yaml.load(open(path), Loader = yaml.SafeLoader)
		# ------------------------------ camera information
		IW = config["IW"]
		IH = config["IH"]
		FX = config["FX"]
		FY = config["FY"]
		CX = config["CX"]
		CY = config["CY"]
		config["roi"] = (-CX, -CY, IW - CX, IH - CY)
		config["focal"] = FX
		config["aspect"] = FX / FY
		# ------------------------------ image pre-process
		config["blur_sigma"] = config.get("blur_sigma", None)
		config["K"] = config.get("K", 1.5)
		config["new_focal"] = FX / config["K"]
		# ------------------------------ feature det setting
		config["use_feature"] = config.get("use_feature", True)
		config["knn_ratio_th"] = config.get("knn_ratio_th", 0.7)
		if config["use_feature"]:
			name = config.get("feature_type", "SIFT").lower()
			feature_dict = {
				"sift": cv2.SIFT_create,
				"surf": cv2.xfeatures2d.SURF_create,
				"brisk": cv2.BRISK_create,
				"akaze": cv2.AKAZE_create,
				"orb": cv2.ORB_create,
			}
			config["detector"] = feature_dict[name]()
			if name in ("orb", "brisk", "akaze"):
				index_params = dict(  # orb, brisk, akaze
					algorithm = 6,  # FLANN_INDEX_LSH
					table_number = 6,  # 6, 12
					key_size = 12,  # 12, 20
					multi_probe_level = 2,  # 1, 2
				)
			else:
				index_params = dict(  # sift, surf
					algorithm = 1,  # FLANN_INDEX_KDTREE
					trees = 5,
				)
			search_params = dict(checks = 100)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			config["matcher"] = flann.knnMatch
		else:
			config["detector"] = None
			config["matcher"] = None
		# ------------------------------ optical flow setting
		config["use_optFlow"] = config.get("use_optFlow", True)
		config["flow_min_dis"] = config.get("flow_min_dis", 10)
		config["flow_win_size"] = config.get("flow_win_size", 30)
		config["flow_level"] = config.get("flow_level", 3)
		# ------------------------------ common setting
		config["distribute_win_size"] = config.get("distribute_win_size", 10)
		# ------------------------------ check
		assert config["use_feature"] or config["use_optFlow"], log.error("at least one type")
		return config

	def __init__(self, image, **config):
		"""
		:param image:
		:param config:
		:return:
		"""
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sigma = config["blur_sigma"]
		if sigma is not None:
			gray = cv2.GaussianBlur(gray, (3, 3), sigma, borderType = cv2.BORDER_REPLICATE)
		self.focal = config["new_focal"]
		gray, mask, roi = remapping(
			gray, None, config["roi"], None, project_flat, recover_flat, config["focal"],
			project_cylinder, recover_cylinder, self.focal, aspect = config["aspect"],
			interpolation = cv2.INTER_CUBIC, borderType = cv2.BORDER_REPLICATE,
		)
		self.gray = gray
		self.mask = mask
		self.roi = roi
		#  ---------------------------------------------------
		self.use_feature = config["use_feature"]
		self.detector = config["detector"]
		self.matcher = config["matcher"]
		self.knn_ratio_th = config["knn_ratio_th"]
		if self.use_feature:
			self.fea = cv2.detail.computeImageFeatures2(self.detector, gray, mask)
		else: self.fea = None
		#  ---------------------------------------------------
		self.use_optFlow = config["use_optFlow"]
		self.flow_min_dis = config["flow_min_dis"]
		self.flow_win_size = config["flow_win_size"]
		self.flow_level = config["flow_level"]
		#  ---------------------------------------------------
		self.distribute_win_size = config["distribute_win_size"]
		pass

	def _match_feature(self, other):
		"""
		:param other:
		:return:
			flag: if succeed
			U0: kpts in self
			U1: kpts in other
		"""
		des0 = self.fea.descriptors
		kpt0 = self.fea.keypoints
		des1 = other.fea.descriptors
		kpt1 = other.fea.keypoints
		# ========= >> match feature descriptors
		matches = self.matcher(des0, des1, k = 2)
		good_matches = []
		for m, n in matches:
			if m.distance < self.knn_ratio_th * n.distance:
				good_matches.append(m)
		# ========= >> result process
		good_matches_count = len(good_matches)
		log.info(f"good feature matches count = {good_matches_count} ..")
		if good_matches_count < MATCH_MIN: return False, None, None

		# recover matched image points from keypoints
		U0 = np.asarray([kpt0[m.queryIdx].pt for m in good_matches], np.float32)
		U1 = np.asarray([kpt1[m.trainIdx].pt for m in good_matches], np.float32)
		# distribute feature keypoints, sorted by 'response'
		h, w = self.mask.shape
		idxes = np.argsort([kpt0[m.queryIdx].response for m in good_matches])
		idxes = idxes[::-1]  # mark: warning, do not forget, however, allways forget ..
		pair_ids = distribute_keypoint(U0, idxes, (w, h), self.distribute_win_size)
		inliers_count = len(pair_ids)
		log.info(f"distribute feature inliers count = {inliers_count} ..")
		if inliers_count < MATCH_MIN: return False, None, None
		# --------------------------------------------------------
		# U0_show = U0[pair_ids]
		# U1_show = U1[pair_ids] + (w, 0)
		# can.draw_points(U0_show, marker = ".", s = 2, color = "orangered")
		# can.draw_points(U1_show, marker = ".", s = 2, color = "orangered")
		# idxes_show = np.argsort(np.random.random(len(U0_show)))[:20]
		# can.draw_line(U0_show[idxes_show[:20]], U1_show[idxes_show[:20]], color = "yellow")
		# --------------------------------------------------------
		U0 = U0[pair_ids] + self.roi[:2]
		U1 = U1[pair_ids] + other.roi[:2]
		P0 = recover_cylinder(U0, self.focal)
		P0[:, 0] /= P0[:, 2]
		P0[:, 1] /= P0[:, 2]
		P1 = recover_cylinder(U1, other.focal)
		P1[:, 0] /= P1[:, 2]
		P1[:, 1] /= P1[:, 2]
		return True, P0[:, :2], P1[:, :2]

	def _match_optFlow(self, other):
		# calc Shi-Tomasi corner
		U0 = cv2.goodFeaturesToTrack(self.gray, 0, 0.01, self.flow_min_dis, mask = self.mask)
		U0 = U0.reshape((-1, 2))
		U0 = np.asarray(U0, np.float32)
		U1, status, err = cv2.calcOpticalFlowPyrLK(
			self.gray, other.gray, U0, None, maxLevel = self.flow_level,
			winSize = (self.flow_win_size, self.flow_win_size),
			flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
		)
		U1 = U1.reshape((-1, 2))
		U1 = np.asarray(U1, np.float32)
		err = err.ravel()
		# check, if U1 in frame
		h, w = self.mask.shape
		valid_w = np.logical_and(U1[:, 0] > 0, U1[:, 0] < w)
		valid_h = np.logical_and(U1[:, 1] > 0, U1[:, 1] < h)
		valid = np.logical_and(valid_w, valid_h)
		good_matches_count = np.sum(valid)
		log.info(f"good optFlow matches count = {good_matches_count} ..")
		if good_matches_count < MATCH_MIN: return False, None, None
		U0 = U0[valid]
		U1 = U1[valid]
		err = err[valid]
		# distribute flow keypoints, sorted err
		idxes = np.argsort(err)
		pair_ids = distribute_keypoint(U0, idxes, (w, h), self.distribute_win_size)
		inliers_count = len(pair_ids)
		log.info(f"distribute optFlow inliers count = {inliers_count} ..")
		if inliers_count < MATCH_MIN: return False, None, None
		# --------------------------------------------------------
		# U0_show = U0[pair_ids]
		# U1_show = U1[pair_ids] + (w, 0)
		# # can.draw_points(U0_show, marker = ".", s = 2, color = "lime")
		# can.draw_points(U1_show, marker = ".", s = 2, color = "lime")
		# idxes_show = np.argsort(np.random.random(len(U0_show)))[:20]
		# can.draw_line(U0_show[idxes_show[:20]], U1_show[idxes_show[:20]], color = "cyan")
		# --------------------------------------------------------
		U0 = U0[pair_ids] + self.roi[:2]
		U1 = U1[pair_ids] + other.roi[:2]
		P0 = recover_cylinder(U0, self.focal)
		P0[:, 0] /= P0[:, 2]
		P0[:, 1] /= P0[:, 2]
		P1 = recover_cylinder(U1, other.focal)
		P1[:, 0] /= P1[:, 2]
		P1[:, 1] /= P1[:, 2]
		return True, P0[:, :2], P1[:, :2]

	def match(self, other):
		"""
		:param other:
		:return:
			flag:
			U0:
			U1:
		"""
		flag = False
		U0 = np.empty((0, 2), np.float32)
		U1 = np.empty((0, 2), np.float32)
		if self.use_feature:
			log.info("running feature matching ..")
			flag_fea, U0_fea, U1_fea = self._match_feature(other)
			if not flag_fea:
				log.critical("feature matching failed ..")
			else:
				flag = True
				U0 = np.concatenate([U0, U0_fea], dtype = np.float32)
				U1 = np.concatenate([U1, U1_fea], dtype = np.float32)
		if self.use_optFlow:
			log.info("running optFlow matching ..")
			flag_opt, U0_opt, U1_opt = self._match_optFlow(other)
			if not flag_opt:
				log.critical("optFlow matching failed ..")
			else:
				flag = True
				U0 = np.concatenate([U0, U0_opt], dtype = np.float32)
				U1 = np.concatenate([U1, U1_opt], dtype = np.float32)
		if flag: return True, U0, U1
		return False, None, None

	def match_roughly(self, other):
		flow = cv2.calcOpticalFlowFarneback(
			self.gray, other.gray, None, 0.5, levels = self.flow_level,
			winsize = self.flow_win_size, iterations = 5, poly_n = 5, poly_sigma = 1.1,
			flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
		)
		scale = 1 / self.distribute_win_size
		sub_flow = cv2.resize(flow, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
		h_, w_ = sub_flow.shape[:2]
		# sub_flow = sub_flow.reshape((-1, 2))
		v, u = np.indices((h_, w_), int)
		u = u.ravel()
		v = v.ravel()
		U0 = np.asarray([u, v], np.float32).T / scale
		dU = sub_flow[v, u]
		# distribute feature keypoints, sorted by U0 - U1 distance
		flow_D2 = np.sum(dU * dU, axis = 1)
		valid = flow_D2 > DISTANCE_MIN2
		inliers_count = np.sum(valid)
		log.info(f"kpts inliers count = {inliers_count} ..")
		if inliers_count < MATCH_MIN: return False, None, None
		dU = dU[valid]
		U0 = U0[valid]
		U1 = U0 + dU
		# --------------------------------------------------------
		# h, w = self.gray.shape[:2]
		# U0_show = U0[valid]
		# U1_show = U0_show + dU + np.asarray((w, 0), np.float32)
		# can.draw_points(U0_show, marker = ".", s = 2, color = "orangered")
		# can.draw_points(U1_show, marker = ".", s = 2, color = "orangered")
		# can.draw_line(U0_show, U1_show, color = "yellow")
		# idxes_show = np.argsort(np.random.random(len(U0_show)))[:20]
		# can.draw_line(U0_show[idxes_show[:20]], U1_show[idxes_show[:20]], color = "yellow")
		# --------------------------------------------------------
		U0 += self.roi[:2]
		U1 += other.roi[:2]
		P0 = recover_cylinder(U0, self.focal)
		P0[:, 0] /= P0[:, 2]
		P0[:, 1] /= P0[:, 2]
		P1 = recover_cylinder(U1, other.focal)
		P1[:, 0] /= P1[:, 2]
		P1[:, 1] /= P1[:, 2]
		return True, P0[:, :2], P1[:, :2]


if __name__ == '__main__':
	from Core.Visualization import SpaceCanvas, KaisCanvas
	from Core.ComputerVision.frame_track import recover_pose
	from Core.ComputerVision.image_io import imread_raw

	KaisLog.set_level("info")

	folder_source = "/home/kai/Desktop/DailySource/__Archived/MMS_PROBE_IMAGE_SETS/"
	folder_i = folder_source + "210519051645/"
	folder_o = "/home/kai/Desktop/DailySource/M23_04/D2304_02/out/"

	can3D = SpaceCanvas(folder_o)
	can = KaisCanvas(fig_size = (12, 6), fig_edge = (0.5, 0.53, 0.2, 0.35), line_width = 0.4)


	def main():
		IW = 1280
		IH = 720
		img_prev = imread_raw(folder_i + "051647296.raw", IW, IH)
		img_curr = imread_raw(folder_i + "051647499.raw", IW, IH)

		cvf_config = CVFrame.get_cvf_config()

		frame1 = CVFrame(img_prev, **cvf_config)
		cv2.imwrite(folder_o + "img_1.jpg", img_prev)
		cv2.imwrite(folder_o + "gray_1.jpg", frame1.gray)
		frame2 = CVFrame(img_curr, **cvf_config)
		cv2.imwrite(folder_o + "img_2.jpg", img_curr)
		cv2.imwrite(folder_o + "gray_2.jpg", frame2.gray)

		# --------------------------------------------------------
		h, w = frame1.gray.shape[:2]
		img_show = np.zeros((h, w + w), np.uint8)
		img_show[:, :w] = frame1.gray
		img_show[:, w:] = frame2.gray
		img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
		can.ax.imshow(img_show, alpha = 0.75)
		# --------------------------------------------------------
		flag, U1, U2 = frame1.match_roughly(frame2)
		flag, R, t, valid = recover_pose(U1, U2)
		print(R, t)

		# U1 = U1 * (FX, FY) - ROI[:2]
		# U2 = U2 * (FX, FY) - ROI[:2]
		# can.draw_line(U1, U2, color = "orange", alpha = 0.7)
		# can.draw_line(U1, U2, color = "orange", alpha = 0.7)
		# --------------------------------------------------------
		can.set_axis(equal_axis = True, grid_on = False)
		can.ax.grid(which = "both", zorder = 0, alpha = 0.2)
		can.ax.minorticks_on()

		# can3D.save(folder_out + "out_featureORB.jpg")
		can.save(folder_o + "fig.jpg")
		# --------------------------------------------------------

		pass


	main()
	can.close()
	# can3D.save()
	pass
pass
