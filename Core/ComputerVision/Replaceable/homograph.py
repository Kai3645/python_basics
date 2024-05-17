import cv2
import numpy as np

from Core.Basic import KaisLog
from Core.ComputerVision import normalize_universal
from Core.ComputerVision.Replaceable import remapping, resize
from Core.ComputerVision.frame_track import distribute_keypoint

log = KaisLog.get_log()


def findHomography(
	gray_1, mask_1, roi_1, project_func_1, recover_func_1, focal_1,
	gray_2, mask_2, roi_2, project_func_2, recover_func_2, focal_2,
	project_func_0, recover_func_0, focal_0, detector, matcher, *,
	aspect_1 = 1, aspect_2 = 1, distribute_step = 20, inliers_th = 50,
	interpolation = cv2.INTER_CUBIC
):
	"""
	require:
		mapping img_2 into img_1
	==>	calc H,  img_2 --> img_1
	memo:
	for func_0, keypoints location
	1) flat <-- evenly distributed
	2) cylinder <-- horizontal ellipse near center <<< working good
	3) stereographic <-- only center area
	:param gray_1:
	:param mask_1:
	:param roi_1:
	:param project_func_1:
	:param recover_func_1:
	:param focal_1:
	:param gray_2:
	:param mask_2:
	:param roi_2:
	:param project_func_2:
	:param recover_func_2:
	:param focal_2:
	:param project_func_0:
	:param recover_func_0:
	:param focal_0:
	:param detector:
	:param matcher:
	:param distribute_step:
	:param aspect_1: input image's fx / fy
	:param aspect_2: input image's fx / fy
	:param inliers_th:
	:param interpolation:
	:return:
		H:
	"""
	if project_func_1 is not None:
		gray_1, mask_1, roi_1 = remapping(gray_1, mask_1, roi_1, None,
		                                  project_func_1, recover_func_1, focal_1,
		                                  project_func_0, recover_func_0, focal_0,
		                                  aspect = aspect_1, interpolation = interpolation,
		                                  borderType = cv2.BORDER_REPLICATE)
	elif focal_0 != focal_1:
		gray_1, mask_1, roi_1 = resize(gray_1, mask_1, roi_1, focal_0 / focal_1, interpolation)
	fea_1 = cv2.detail.computeImageFeatures2(detector, gray_1, mask_1)
	if len(fea_1.keypoints) < inliers_th: return None

	if project_func_2 is not None:
		gray_2, mask_2, roi_2 = remapping(gray_2, mask_2, roi_2, None,
		                                  project_func_2, recover_func_2, focal_2,
		                                  project_func_0, recover_func_0, focal_0,
		                                  aspect = aspect_2, interpolation = interpolation,
		                                  borderType = cv2.BORDER_REPLICATE)
	elif focal_0 != focal_2:
		gray_2, mask_2, roi_2 = resize(gray_2, mask_2, roi_2, focal_0 / focal_2, interpolation)
	fea_2 = cv2.detail.computeImageFeatures2(detector, gray_2, mask_2)
	if len(fea_2.keypoints) < inliers_th: return None

	pairs = matcher.apply(fea_1, fea_2)
	log.info(f"match inliers = {pairs.num_inliers} ..")
	if pairs.num_inliers < inliers_th: return None

	matches = pairs.getMatches()
	idxes = np.where(pairs.getInliers() > 0)[0]
	U1 = np.asarray([fea_1.keypoints[matches[i].queryIdx].pt for i in idxes], np.float32)
	U2 = np.asarray([fea_2.keypoints[matches[i].trainIdx].pt for i in idxes], np.float32)

	h, w = mask_2.shape
	idxes = np.argsort([fea_2.keypoints[matches[i].trainIdx].response for i in idxes])
	idxes = idxes[::-1]  # mark: warning, do not forget, however, allways forget ..
	pair_ids = distribute_keypoint(U2, idxes, (w, h), distribute_step)
	num_inliers = len(pair_ids)
	log.info(f"distribute inliers = {num_inliers} ..")
	if num_inliers < inliers_th: return None
	# mark: warning, + 0.5 will make result bad ..
	U1 = U1[pair_ids] + roi_1[:2]  # + 0.5
	U2 = U2[pair_ids] + roi_2[:2]  # + 0.5

	P1 = recover_func_0(U1, focal_0)
	P1[:, 0] /= P1[:, 2]
	P1[:, 1] /= P1[:, 2]
	P2 = recover_func_0(U2, focal_0)
	P2[:, 0] /= P2[:, 2]
	P2[:, 1] /= P2[:, 2]

	U1_, K1, inv_K1 = normalize_universal(P1[:, :2])
	U2_, K2, inv_K2 = normalize_universal(P2[:, :2])
	H_, _ = cv2.findHomography(U2_, U1_, method = cv2.RANSAC)
	H = inv_K1.dot(H_).dot(K2)
	return H
