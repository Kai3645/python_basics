import cv2
import numpy as np

inter_dict = {
	"nearest": cv2.INTER_NEAREST,
	"area": cv2.INTER_AREA,
	"linear": cv2.INTER_LINEAR,
	"linear_exact": cv2.INTER_LINEAR_EXACT,
	"cubic": cv2.INTER_CUBIC,
	"lanczos4": cv2.INTER_LANCZOS4,
}
features_dict = {
	"sift": cv2.SIFT_create,
	"surf": cv2.xfeatures2d.SURF_create,
	"brisk": cv2.BRISK_create,
	"akaze": cv2.AKAZE_create,
	"orb": cv2.ORB.create,
}
estimator_dict = {
	"homography": cv2.detail_HomographyBasedEstimator,
	"affine": cv2.detail_AffineBasedEstimator,
}
ba_cost_dict = {
	"ray": cv2.detail_BundleAdjusterRay,
	"reproj": cv2.detail_BundleAdjusterReproj,
	"affine": cv2.detail_BundleAdjusterAffinePartial,
	"no": cv2.detail_NoBundleAdjuster,
}
expos_comp_dict = {  # Exposure compensation method.
	"gain_blocks": cv2.detail.ExposureCompensator_GAIN_BLOCKS,
	"gain": cv2.detail.ExposureCompensator_GAIN,
	"channel": cv2.detail.ExposureCompensator_CHANNELS,
	"channel_blocks": cv2.detail.ExposureCompensator_CHANNELS_BLOCKS,
	"no": cv2.detail.ExposureCompensator_NO,
}
seam_find_dict = {  # Seam estimation method.
	"gc_color": cv2.detail_GraphCutSeamFinder('COST_COLOR'),
	"gc_colorgrad": cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD'),
	"dp_color": cv2.detail_DpSeamFinder('COLOR'),
	"dp_colorgrad": cv2.detail_DpSeamFinder('COLOR_GRAD'),
	"voronoi": cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM),
	"no": cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO),
}


def set_scales(work_pix, seam_pix, sam_shape):
	area = sam_shape[0] * sam_shape[1]
	work_scale = min(1.0, np.sqrt(work_pix * 1e6 / area))
	seam_scale = min(1.0, np.sqrt(seam_pix * 1e6 / area))
	aspect_s_w = seam_scale / work_scale
	return work_scale, seam_scale, aspect_s_w


def get_matcher(matcher_type, feature_type, match_conf):
	try_gpu = True
	if match_conf is None:
		if feature_type == 'orb': match_conf = 0.3
		else: match_conf = 0.65
	if matcher_type == "affine":
		matcher = cv2.detail_AffineBestOf2NearestMatcher(False, try_gpu, match_conf)
	else: matcher = cv2.detail.BestOf2NearestMatcher_create(try_gpu, match_conf)
	return matcher


def get_compensator(num_feed, block_size, num_filter, expos_comp):
	if expos_comp == cv2.detail.ExposureCompensator_CHANNELS:
		compensator = cv2.detail_ChannelsCompensator(num_feed)
	elif expos_comp == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
		compensator = cv2.detail_BlocksChannelsCompensator(block_size, block_size, num_feed)
		compensator.setNrGainsFilteringIterations(num_filter)
	else: compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp)
	return compensator


def get_rois(cameras, sizes, aspect, warper):
	rois = np.zeros((len(sizes), 4), np.int)
	for i, (cam, s) in enumerate(zip(cameras, sizes)):
		cam.focal *= aspect
		cam.ppx *= aspect
		cam.ppy *= aspect
		K = cam.K().astype(np.float32)
		rois[i, :] = warper.warpRoi(tuple(s), K, cam.R)
	rois[:, :2] -= np.min(rois[:, :2], axis = 0)
	return rois


def set_blender(rois: np.ndarray, blend_type, strength):
	dst_roi = np.zeros(4, np.int)
	dst_roi[:2] = np.min(rois[:, :2], axis = 0)
	dst_roi[2:] = np.max(rois[:, :2] + rois[:, 2:], axis = 0)

	blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * strength / 100
	blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
	if blend_width < 1: pass
	elif blend_type == "multiband":
		blender = cv2.detail_MultiBandBlender()
		blender.setNumBands((np.log(blend_width) / np.log(2) - 1).astype(np.int))
	elif blend_type == "feather":
		blender = cv2.detail_FeatherBlender()
		blender.setSharpness(1 / blend_width)
	blender.prepare(dst_roi)
	return blender


def show_camera_info(cameras):
	count = len(cameras)
	K = np.zeros((count, 3, 3))
	for i, cam in enumerate(cameras):
		K[i, :, :] = cam.K()
		print(f"camera mat @ {i} =\n{cam.R}")
	ave_K = np.mean(K, axis = 0).round(1)
	print(f"camera matrix = \n{ave_K}")
	pass


def stitching(src_imgs, folder, **kwargs):
	"""
	:param src_imgs: image array
	:param folder: folder to save step images for debug
	:param kwargs:
		work_pix: Resolution for image registration step. default 1.0 Mpx
		seam_pix: Resolution for seam estimation step. default 0.2 Mpx
		feature_type: {"surf", "sift", "brisk", "akaze", "orb"}
		match_conf: default None
		matcher_type: {"homography", "affine"}
		ba_cost_type: {"ray", "reproj", "affine", "no"}
		ba_refine_mask: Set refinement mask for bundle adjustment. default "yyyyy"
                        where "y" means refine respective parameter and "n" means don't refine,
                        and has the following format: "fx", "skew", "ppx", "aspect", "ppy".
		wave_correct: Perform wave effect correction. {"horiz", "vert", "no"}
		warp_type: {"plane", "spherical", "affine", "cylindrical", "fisheye",
					"stereographic", "compressedPlaneA2B1", "compressedPlaneA1.5B1",
					"compressedPlanePortraitA2B1", "compressedPlanePortraitA1.5B1",
					"paniniA2B1", "paniniA1.5B1", "paniniPortraitA2B1",
					"paniniPortraitA1.5B1", "mercator", "transverseMercator"}
		comp_type: {"gain_blocks", "gain", "channel", "channel_blocks", "no"}
		comp_num_feed: Number of exposure compensation feed. default 1
		seam_find_type: {"gc_color", "gc_colorgrad", "dp_color", "dp_colorgrad", "voronoi", "no"}
		blend_type: Blending method. {"multiband", "feather", "no"}
		blend_strength: Blending strength from [0,100] range. default 5
		comp_block_size: BLock size in pixels used by the exposure compensator. default 32
		comp_num_filter: Number of filtering iterations of the exposure compensation gains. default 2
	:return: result image
	"""
	print(">> process start .. ")
	if folder[-1] != "/": folder += "/"
	sam_h, sam_w = src_imgs[0].shape[:2]  # sample shape

	work_pix = kwargs.get("work_pix", 1.0)
	seam_pix = kwargs.get("seam_pix", 0.2)
	work_scale, seam_scale, aspect_s_w = set_scales(work_pix, seam_pix, (sam_w, sam_h))
	print("----------------------------------------")
	print(f"work scale = {work_scale}")
	print(f"seam scale = {seam_scale}")
	print("----------------------------------------")

	work_inter = kwargs.get("work_inter", "linear_exact")
	seam_inter = kwargs.get("seam_inter", "linear_exact")

	feature_type = kwargs.get("feature_type", "surf")
	finder = features_dict[feature_type]()

	count = len(src_imgs)
	src_sizes = np.zeros((count, 2), np.int)
	seam_imgs = np.empty(count, object)
	features = np.empty(count, object)

	print(">> resizing .. ")
	for i, img in enumerate(src_imgs):
		h, w = img.shape[:2]
		src_sizes[i, :] = w, h
		tmp = cv2.resize(
			img, None,
			fx = work_scale,
			fy = work_scale,
			interpolation = inter_dict[work_inter]
		)
		seam_imgs[i] = cv2.resize(
			img, None,
			fx = seam_scale,
			fy = seam_scale,
			interpolation = inter_dict[seam_inter]
		)
		features[i] = cv2.detail.computeImageFeatures2(finder, tmp)

	print(">> matching part 1 .. ")
	match_conf = kwargs.get("match_conf", None)
	matcher_type = kwargs.get("matcher_type", "homography")
	matcher = get_matcher(matcher_type, feature_type, match_conf)
	p = matcher.apply2(features)
	matcher.collectGarbage()
	idxes = cv2.detail.leaveBiggestComponent(features, p, 0.3)
	idxes = np.asarray(idxes).ravel()

	count = len(idxes)
	print(">> matching part 2 .. ")
	assert count >= 2, "Need more images .. "
	src_sizes = src_sizes[idxes]
	src_imgs = src_imgs[idxes]
	seam_imgs = seam_imgs[idxes]
	features = features[idxes]
	p = matcher.apply2(features)
	print("----------------------------------------")
	print(f"matched indexes = {idxes}")
	print(f"total matched = {count}")
	print("----------------------------------------")

	default_cameraPara = kwargs.get("default_cameraPara", None)
	if default_cameraPara is not None:
		print(">> creating camera matrix ..")
		cameras = [cv2.detail_CameraParams() for _ in range(count)]
		for cam in cameras:
			cam.aspect = default_cameraPara.aspect
			cam.focal = default_cameraPara.focal * work_scale
			cam.ppx = default_cameraPara.ppx * work_scale
			cam.ppy = default_cameraPara.ppy * work_scale
			cam.R = np.eye(3, dtype = np.float32)
			cam.t = np.zeros((3, 1), np.float32)
	else:
		print(">> estimating camera matrix ..")
		estimator = estimator_dict[matcher_type]()
		b, cameras = estimator.apply(features, p, None)
		assert b, "Homography estimation failed .. "
		for cam in cameras: cam.R = cam.R.astype(np.float32)

	print(">> bundle adjustment for camera matrix .. ")
	ba_cost_type = kwargs.get("ba_cost_type", "ray")
	ba_refine_mask = kwargs.get("ba_refine_mask", "yyyyy")
	adjuster = ba_cost_dict[ba_cost_type]()
	adjuster.setConfThresh(1)
	refine_mask = np.zeros((3, 3), np.uint8)
	if ba_refine_mask[0] == "y": refine_mask[0, 0] = 1
	if ba_refine_mask[1] == "y": refine_mask[0, 1] = 1
	if ba_refine_mask[2] == "y": refine_mask[0, 2] = 1
	if ba_refine_mask[3] == "y": refine_mask[1, 1] = 1
	if ba_refine_mask[4] == "y": refine_mask[1, 2] = 1
	adjuster.setRefinementMask(refine_mask)
	b, cameras = adjuster.apply(features, p, cameras)
	assert b, "Camera parameters adjusting failed."
	warped_scale = np.median([cam.focal for cam in cameras])

	print(">> wave correcting .. ")
	wave_correct = kwargs.get("wave_correct", "horiz")
	if wave_correct == 'horiz' or wave_correct == "vert":
		rmats = [np.copy(cam.R) for cam in cameras]
		if wave_correct == "horiz":
			rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
		else: rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_VERT)
		for i, cam in enumerate(cameras): cam.R = rmats[i]

	masks = np.empty(count, object)
	for i, img in enumerate(seam_imgs):
		h, w = img.shape[:2]
		masks[i] = cv2.UMat(255 * np.ones((h, w), np.uint8))

	warp_type = kwargs.get("warp_type", "plane")
	warper = cv2.PyRotationWarper(warp_type, warped_scale * aspect_s_w)  # warper could be nullptr?
	warp_inter = kwargs.get("warp_inter", "linear")

	print(">> wrapping .. ")
	corners = []
	imgs_warped = []
	masks_warped = []
	for i, (img, msk, cam) in enumerate(zip(seam_imgs, masks, cameras)):
		K = cam.K().astype(np.float32)
		K[0, 0] *= aspect_s_w
		K[0, 2] *= aspect_s_w
		K[1, 1] *= aspect_s_w
		K[1, 2] *= aspect_s_w
		corn, img_wp = warper.warp(img, K, cam.R, inter_dict[warp_inter], cv2.BORDER_REFLECT)
		corners.append(corn)
		imgs_warped.append(img_wp)
		_, mask_wp = warper.warp(msk, K, cam.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
		masks_warped.append(mask_wp.get())
	imgs_warped_f = [img.astype(np.float32) for img in imgs_warped]

	print(">> compensating .. ")
	comp_type = kwargs.get("comp_type", "gain_blocks")
	compensator = get_compensator(
		num_feed = kwargs.get("comp_num_feed", 1),
		block_size = kwargs.get("comp_block_size", 32),
		num_filter = kwargs.get("comp_num_filter", 2),
		expos_comp = expos_comp_dict[comp_type],
	)
	compensator.feed(corners, imgs_warped, masks_warped)

	print(">> seam estimating .. ")
	seam_find_type = kwargs.get("seam_find_type", "gc_color")
	seam_finder = seam_find_dict[seam_find_type]
	seam_finder.find(imgs_warped_f, corners, masks_warped)

	print(">> calculating roi .. ")
	aspect_compose = 1 / work_scale
	warped_scale *= aspect_compose
	warper = cv2.PyRotationWarper(warp_type, warped_scale)
	rois = get_rois(cameras, src_sizes, aspect_compose, warper)

	blend_type = kwargs.get("blend_type", "multiband")
	blend_strength = kwargs.get("blend_strength", 5)
	blender = set_blender(rois, blend_type, blend_strength)

	print("----------------------------------------")
	print(f"rois = \n{rois}")
	show_camera_info(cameras)
	print("----------------------------------------")

	print(">> blending: ")
	blend_inter = kwargs.get("blend_inter", "cubic")
	for i, (img, cam) in enumerate(zip(src_imgs, cameras)):
		K = cam.K().astype(np.float32)
		_, img_wp = warper.warp(img, K, cam.R, inter_dict[blend_inter], cv2.BORDER_REFLECT)
		msk = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
		_, msk_wp = warper.warp(msk, K, cam.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
		compensator.apply(i, tuple(rois[i, :2]), img_wp, msk_wp)
		img_wp_s = img_wp.astype(np.int16)
		cv2.imwrite(folder + "warped_%03d.png" % i, cv2.bitwise_and(img_wp, img_wp, mask = msk_wp))
		blender.feed(cv2.UMat(img_wp_s), msk_wp, tuple(rois[i, :2]))
	result, result_mask = blender.blend(None, None)
	h, w = result.shape[:2]
	print("----------------------------------------")
	print(f"result image shape = ({w}, {h})")
	print("----------------------------------------")
	print(">> process finished .. ")
	return result


"""
if __name__ == '__main__':
	# a simple sample
	import os

	from tqdm import tqdm

	main_folder = ""
	folder_input = main_folder + "sample"
	folder_output = main_folder + "output"

	img_type = ".png"
	paths = os.listdir(folder_input)
	src_images = []
	for f in tqdm(paths, desc = ">> scanning "):
		if img_type not in f: continue
		image = cv2.imread(folder_input + f)
		if image is None: err_exit(f"can not read image {f} .. ")
		src_images.append(image)
	src_images = np.asarray(src_images)

	cameraPara = cv2.detail_CameraParams()
	cameraPara.aspect = 1.0
	cameraPara.focal = 3271.9
	cameraPara.ppx = 1981.55
	cameraPara.ppy = 1472.71

	dst = stitching(
		src_imgs = src_images,
		folder = folder_output,
		default_cameraPara = cameraPara,
		work_pix = 1.5, seam_pix = 0.5,
		feature_type = "surf",
		work_inter = "linear_exact",
		ba_cost_type = "reproj",
		ba_refine_mask = "yyyny",
		wave_correct = "horiz",
		warp_type = "cylindrical",
		comp_type = "gain_blocks",
		comp_block_size = 100,
		comp_num_filter = 2,
		blend_inter = "cubic",
		blend_strength = 5,
	)
	if dst is None: err_exit(f"can not stitching image .. ")
	cv2.imwrite(folder_output + "result.jpg", dst)
	pass
"""
