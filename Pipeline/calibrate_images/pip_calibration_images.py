"""
usage: python "pip_video2image.py" "config.yaml"
"""

if __name__ == '__main__':
	import os
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

	import yaml
	import time

	print(">> loading .. ")
	config_path = sys.argv[1]
	config = yaml.load(open(config_path), Loader = yaml.SafeLoader)
	time_start = time.time()

	folder_out = config["folder_out"]
	images_path = config["images_path"]
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
	calib_test_num = config["calib_test_num"]
	calib_mse_err_th = config["calib_mse_err_th"]
	calib_max_err_th = config["calib_max_err_th"]
	change_th = IMAGE_W * IMAGE_H * config["CHANGE_RATIO"]
	do_rotate_image = config["ROTATE_IMAGE"]

	import cv2
	import numpy as np
	from tqdm import tqdm

	from Core.Basic import mkdir, str2folder, listdir, err_exit
	from Core.ComputerVision.camera_basic import chessboard_corner_detect, chessboard_area_ratio, \
		calibrate_camera, calibrate_demo_image, chessboard_draw_corner, chessboard_corner_3d
	from Core.ComputerVision.image_basic import gray2binary

	folder_out = str2folder(folder_out)
	images_path = str2folder(images_path)

	# collect images paths
	img_names = listdir(images_path, pattern = "*.jpg")

	folder_source = mkdir(folder_out, "source")
	folder_boards = mkdir(folder_out, "boards")
	folder_undist = mkdir(folder_out, "undist")

	img_corners = []
	binary_old = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
	for i, name in enumerate(tqdm(img_names, desc = ">> scanning .. ")):
		img = cv2.imread(images_path + name)
		if img is None:
			err_exit("can not read image, @\"" + name + "\" .. ")
		img_shape = img.shape
		if img_shape[0] != IMAGE_H or img_shape[1] != IMAGE_W:
			err_exit("image shape error, @\"" + name + "\" .. ")

		if do_rotate_image: img = img[::-1, ::-1]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		binary = gray2binary(gray)

		diff_valid = binary_old != binary
		if np.sum(diff_valid) < change_th: continue

		corners = chessboard_corner_detect(gray, board_shape, binary)
		if corners is None: continue

		area_ratio = chessboard_area_ratio(corners, board_shape, image_shape)
		if area_ratio < AREA_RATIO_MIN or area_ratio > AREA_RATIO_MAX: continue

		cv2.imwrite(folder_source + f"{i:04d}.jpg", img)
		dst = chessboard_draw_corner(img, corners, cname = "hsv")
		cv2.imwrite(folder_boards + f"{i:04d}.jpg", dst)

		img_corners.append(corners)
		binary_old = binary
	img_corners = np.asarray(img_corners)
	print(f"inlier num = {len(img_corners)}")

	# calibrate
	if calib_model < 0:
		calib_flags = (cv2.CALIB_FIX_K1 |
		               cv2.CALIB_FIX_K2 |
		               cv2.CALIB_FIX_K3 |
		               cv2.CALIB_ZERO_TANGENT_DIST)
	else:
		calib_flags = 0
		if calib_model > 0: calib_flags |= cv2.CALIB_RATIONAL_MODEL
		if calib_model > 1: calib_flags |= cv2.CALIB_TILTED_MODEL
		if calib_model > 2: calib_flags |= cv2.CALIB_THIN_PRISM_MODEL

	mat, dist = calibrate_camera(
		img_corners, BOARD_W, BOARD_H, BOARD_R, image_shape, calib_test_num,
		calib_flags = calib_flags, info_folder = folder_out,
		mse_err_th = calib_mse_err_th, max_err_th = calib_max_err_th,
	)
	mat = mat.round(4)
	dist = dist.round(8)

	# undistort sample
	demo_image = calibrate_demo_image(image_shape)
	cv2.imwrite(folder_out + "undistort_sample.jpg", demo_image)
	new_mat, roi = cv2.getOptimalNewCameraMatrix(mat, dist, image_shape, 1)
	dst = cv2.undistort(demo_image, mat, dist, None, new_mat)
	cv2.imwrite(folder_out + "undistort_result_1.jpg", dst)
	dst = cv2.undistort(demo_image, mat, dist)
	cv2.imwrite(folder_out + "undistort_result_2.jpg", dst)

	img_names = listdir(images_path)
	for name in tqdm(img_names, desc = ">> undistorting .. "):
		img = cv2.imread(images_path + name)
		if do_rotate_image: img = img[::-1, ::-1]

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners = chessboard_corner_detect(gray, board_shape)
		if corners is None: continue

		corners_3d = chessboard_corner_3d(BOARD_W, BOARD_H, BOARD_R)
		_, rvec, tvec = cv2.solvePnP(corners_3d, corners, mat, dist, flags = cv2.SOLVEPNP_SQPNP)
		reproj, _ = cv2.projectPoints(corners_3d, rvec, tvec, new_mat, None)
		reproj = reproj.reshape((-1, 2))

		dst = cv2.undistort(img, mat, dist, None, new_mat)
		dst = chessboard_draw_corner(dst, reproj, cname = "cool", center_only = True)

		cv2.imwrite(folder_undist + name, dst)
	time_used = time.time() - time_start
	print(f">> calibration fished, time_used = {time_used:.2f} s .. ")
	print()
	pass
