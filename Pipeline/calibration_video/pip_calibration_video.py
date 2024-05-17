"""
usage:
python \
	"/home/kai/Documents/GitHub/ACRS_MMS/Python/Pipeline/calibration_video/pip_calibration_video.py" \
	"/home/kai/Documents/GitHub/ACRS_MMS/Python/Pipeline/calibration_video/sample_config.yaml"
"""

if __name__ == '__main__':
	import os
	import sys
	import yaml

	print(">> loading info .. ")
	config_path = sys.argv[1]
	config = yaml.load(open(config_path), Loader = yaml.SafeLoader)
	sys.path.append(config["module_python_path"])

	folder_out = config["folder_out"]
	video_path = config["video_path"]
	BOARD_R = config["BOARD_R"]
	BOARD_W = config["BOARD_W"]
	BOARD_H = config["BOARD_H"]
	IMAGE_W = config["IMAGE_W"]
	IMAGE_H = config["IMAGE_H"]
	board_shape = (BOARD_W, BOARD_H)
	image_shape = (IMAGE_W, IMAGE_H)
	AREA_RATIO_MIN = config["AREA_RATIO_MIN"]
	AREA_RATIO_MAX = config["AREA_RATIO_MAX"]
	calib_model = config["calib_model"]

	import cv2
	import numpy as np
	from tqdm import tqdm

	from Core.Basic import KaisLog, mkdir
	from Core.ComputerVision.basic_image import gray2binary
	from Core.ComputerVision.calibration import chessboard_corner_detect, chessboard_area_ratio, calibrate_camera, \
		calibrate_demo_image, chessboard_draw_corner

	KaisLog.set_level("info")
	log = KaisLog.get_log()
	folder_out += os.sep

	# convert video to image
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		log.error(f">> err, can not open video @\"{video_path}\"")
		exit(-1)

	flag, img = cap.read()
	if not flag: exit(-1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if np.sum(gray > 128) < IMAGE_W * IMAGE_H / 4:
		do_inv_image = True
	else: do_inv_image = False

	# folder_source = mkdir(folder_out, "source")
	folder_boards = mkdir(folder_out, "boards")

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	change_th = IMAGE_W * IMAGE_H * 0.1  # if diff area < change_th -> same image
	white_area_th = IMAGE_W * IMAGE_H / 2

	sample_count = 0
	img_corners = []
	binary_old = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
	for i in range(frame_count):
		flag, img = cap.read()
		if not flag: break

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		binary = gray2binary(gray)

		diff_valid = binary_old != binary
		if np.sum(diff_valid) < change_th: continue
		log.info(f"scanning .. {i} -> {i / frame_count * 100:.1f} %")

		corners = chessboard_corner_detect(gray, board_shape, binary)
		if corners is None:
			log.info("can not find corners ..")
			continue

		area_ratio = chessboard_area_ratio(corners, board_shape, image_shape)
		if area_ratio < AREA_RATIO_MIN or area_ratio > AREA_RATIO_MAX:
			log.info(f"over area ratio .. area_ratio = {area_ratio:.4f}")
			continue

		if sample_count % 10 == 0:
			dst = chessboard_draw_corner(img, corners, cname = "hsv")
			cv2.imwrite(folder_boards + f"{sample_count:04d}.jpg", dst)
		sample_count += 1

		img_corners.append(corners)
		binary_old = binary

	cap.release()
	img_corners = np.asarray(img_corners, np.float32)
	log.info("scanning finished ..")

	# calibrate
	if calib_model < 0:
		calib_flags = (cv2.CALIB_FIX_K1 |
		               cv2.CALIB_FIX_K2 |
		               cv2.CALIB_FIX_K3 |
		               cv2.CALIB_ZERO_TANGENT_DIST)
	else:
		calib_flags = 0
		if calib_model > 0: calib_flags |= cv2.CALIB_RATIONAL_MODEL
		if calib_model > 1: calib_flags |= cv2.CALIB_THIN_PRISM_MODEL
		if calib_model > 2: calib_flags |= cv2.CALIB_TILTED_MODEL
	log.info("calibrating start ..")
	mat, dist = calibrate_camera(
		img_corners, BOARD_W, BOARD_H, BOARD_R, image_shape,
		calib_flags = calib_flags, info_folder = folder_out,
	)
	log.info("calibrating finished ..")

	# undistort sample
	demo_image = calibrate_demo_image(image_shape)
	cv2.imwrite(folder_out + "undistort_sample.jpg", demo_image)
	new_mat, roi = cv2.getOptimalNewCameraMatrix(mat, dist, image_shape, 1)
	dst = cv2.undistort(demo_image, mat, dist, None, new_mat)
	cv2.imwrite(folder_out + "undistort_result.jpg", dst)

	folder_undist = mkdir(folder_out, "undistort")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		log.error(f">> err, can not open video @\"{video_path}\"")
		exit(-1)
	log.info("undistorting start ..")
	sample_count = 0
	for i in tqdm(range(frame_count), desc = ">> undistorting .. "):
		flag, img = cap.read()
		if not flag: break
		if sample_count % 40 == 0:
			dst = cv2.undistort(img, mat, dist, None, new_mat)
			cv2.imwrite(folder_undist + f"{sample_count:04d}.jpg", dst)
		sample_count += 1
	log.info("undistorting fished ..")
	log.info("calibration fished .. ")
	pass
