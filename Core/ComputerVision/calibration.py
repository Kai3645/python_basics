import math

import cv2
import numpy as np

from Core.Basic import num2int, num2str, KaisLog
from Core.ComputerVision.basic_image import gray2binary
from Core.Visualization import KaisCanvas, KaisColor

log = KaisLog.get_log()


def chessboard_image(image_width, W, H, is_white_corner = False):
	"""
	BOARD_W = W * 2 + 1
	BOARD_H = H * 2 + 1
	+-----------------------+
	| B - B - B - B - B - B |
	| - B - B - B - B - B - |
	| B - B - B - B - B - B |
	| - B - B - B - B - B - |
	| B - B - B - B - B - B |
	+-----------------------+
	for A3:
		W = 3, H = 2, image_width = 5400
	:param image_width:
	:param W:
	:param H:
	:param is_white_corner:
	:return:
	"""
	BOARD_W = W + W + 1
	BOARD_H = H + H + 1
	rate = int(image_width / BOARD_W)

	if is_white_corner:
		W, B = 1, 0
	else:
		W, B = 0, 1
	board = np.zeros((BOARD_H, BOARD_W), np.uint8)
	board[W::2, B::2] = 255
	board[B::2, W::2] = 255

	re_w = int(BOARD_W * rate)
	re_h = int(BOARD_H * rate)
	board = cv2.resize(board, (re_w, re_h), interpolation = cv2.INTER_NEAREST)
	return board


def chessboard_corner_detect(gray, corner_size, binary = None):
	"""
	:param gray:
	:param corner_size: @opencv: Number of inner corners per a chessboard row and column
	:param binary:
	:return:
	"""
	if binary is None: binary = gray2binary(gray)
	flag, corners = cv2.findChessboardCorners(binary, corner_size, flags = cv2.CALIB_CB_FAST_CHECK)
	if flag is False: return None
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001)
	corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
	corners = corners.reshape((-1, 2))
	if corners[0, 0] < corners[-1, 0]: corners = corners[::-1]
	return corners


def chessboard_draw_corner(src, corners, *, cname: str = "random", center_only: bool = False):
	dst = np.uint8(src * 0.7)
	corners = num2int(corners)
	if cname == "random": colors = KaisColor.rand(length = len(corners))
	else: colors = KaisColor.plotColorMap(cname, length = len(corners), is_gbr = True)

	for uvi, c in zip(corners, colors):
		cv2.circle(dst, tuple(uvi), 1, c, thickness = 2)
		if center_only: continue
		cv2.drawMarker(dst, tuple(uvi), c, markerType = cv2.MARKER_SQUARE, markerSize = 30, thickness = 2)
	for i, c in enumerate(KaisColor.plotColor(KaisColor.axis_cnames, gbr = True)):
		if not center_only: cv2.line(dst, tuple(corners[i]), tuple(corners[i + 1]), c, thickness = 3)
		else: cv2.drawMarker(dst, tuple(corners[i]), c, markerType = cv2.MARKER_SQUARE, markerSize = 30, thickness = 2)
	return dst


def chessboard_area_ratio(corners, border_shape, image_size):
	v1 = corners[-1] - corners[0]
	v2 = corners[border_shape[0] - 1] - corners[border_shape[0] * (border_shape[1] - 1)]
	return 0.5 * abs(np.cross(v1, v2)) / image_size[0] / image_size[1]


def chessboard_corner_3d(border_w, border_h, border_r):
	"""
	:param border_w: width
	:param border_h: height
	:param border_r: distance between each 2 corners [mm]
	:return:
	"""
	points = np.zeros((border_w * border_h, 3), np.float32)
	corners = np.mgrid[:border_h, :border_w].T
	corners = corners.swapaxes(0, 1)
	corners = corners.reshape((-1, 2))
	points[:, :2] = corners[::-1]
	points[:, 0] -= (border_h - 1) / 2
	points[:, 1] -= (border_w - 1) / 2
	return points * border_r


def calibrate_demo_image(image_shape, pixel = 20):
	w, h = image_shape
	img = np.full((h, w), 255, dtype = np.uint8)
	start = pixel // 2
	img[start::pixel, :] = 0
	img[:, start::pixel] = 0
	return img


def calibrate_camera(corners, border_w, border_h, board_r, image_shape, test_num = -1,
                     *, calib_flags, info_folder):
	CALIB_MAX_ABILITY = 200  # take about 40 min
	CALIB_MIN_IMAGE_NUM = 40
	if test_num < 0: test_num = CALIB_MAX_ABILITY
	else: test_num = min(test_num, CALIB_MAX_ABILITY)

	working_corners = corners
	frame_count = len(working_corners)
	assert frame_count >= CALIB_MIN_IMAGE_NUM, ">> need more good chessboard images .. "
	if frame_count > test_num:
		indexes = np.arange(test_num) / test_num
		indexes = (indexes * frame_count).astype(int)
		working_corners = corners[indexes]
		frame_count = test_num

	corners_3d = chessboard_corner_3d(border_w, border_h, board_r)
	object_points = np.tile(corners_3d, (frame_count, 1, 1))
	fw = open(info_folder + "calib_result.txt", "w")
	fw.write(f"input image size = {image_shape[0]}, {image_shape[1]}\n")
	fw.write("--> prepare calib start \n")
	_, mat, dist, _, _ = cv2.calibrateCamera(
		object_points, working_corners, image_shape, None, None,
		flags = calib_flags,
	)
	fw.write("distCoeffs = \n\t" + num2str(dist, 8) + "\n")
	fw.write("camera mat = \n\t" + num2str(mat, 4) + "\n")
	fw.close()

	# compute reproj_err for corners
	corners_count = len(corners)
	Oct7i = int(corners_count / 8 * 7)
	mse_errs = np.zeros(corners_count)
	max_errs = np.zeros(corners_count)
	for i, cs in enumerate(corners):
		# todo: find a fast and accurate solvePnP flag
		_, rvec, tvec = cv2.solvePnP(corners_3d, cs, mat, dist, flags = cv2.SOLVEPNP_IPPE)
		reproj, _ = cv2.projectPoints(corners_3d, rvec, tvec, mat, dist)
		reproj = reproj.reshape((-1, 2))
		err = np.linalg.norm(reproj - cs, axis = 1)
		mse_errs[i] = math.sqrt(np.dot(err, err) / len(cs))
		max_errs[i] = max(err)

	canvas = KaisCanvas(fig_edge = (0.5, 0.5, 0.3, 0.4), linewidth = 1.2)
	ax2 = canvas.ax.twinx()
	idxes_sort = np.argsort(mse_errs)
	canvas.ax.plot(mse_errs[idxes_sort], c = "limegreen", label = "mse_err_1st", zorder = 10)
	ax2.plot(max_errs[idxes_sort], c = "cornflowerblue", lw = 0.6, label = "max_err_1st", zorder = 1)

	max_err_th = max_errs[np.argsort(max_errs)[Oct7i]]
	mse_err_th = mse_errs[np.argsort(mse_errs)[Oct7i]]
	valid = np.logical_and(max_errs < max_err_th, mse_errs < mse_err_th)
	working_corners = corners[valid]
	frame_count = len(working_corners)
	if frame_count > test_num:
		indexes = np.arange(test_num) / test_num
		indexes = (indexes * frame_count).astype(int)
		working_corners = working_corners[indexes]
		frame_count = test_num

	fw = open(info_folder + "calib_result.txt", "a")
	fw.write(f"input board shape = {border_w}, {border_h}\n")
	if frame_count < CALIB_MIN_IMAGE_NUM:
		fw.write("--> fix calib passed \n")
	else:
		object_points = np.tile(corners_3d, (frame_count, 1, 1))
		fw.write("--> fix calib start \n")
		_, mat, dist, _, _ = cv2.calibrateCamera(
			object_points, working_corners, image_shape, None, None,
			flags = calib_flags,
		)
		fw.write("distCoeffs = \n\t" + num2str(dist, 8) + "\n")
		fw.write("camera mat = \n\t" + num2str(mat, 4) + "\n")

		for i, cs in enumerate(corners):
			_, rvec, tvec = cv2.solvePnP(corners_3d, cs, mat, dist, flags = cv2.SOLVEPNP_IPPE)
			reproj, _ = cv2.projectPoints(corners_3d, rvec, tvec, mat, dist)
			reproj = reproj.reshape((-1, 2))
			err = np.linalg.norm(reproj - cs, axis = 1)
			mse_errs[i] = math.sqrt(np.dot(err, err) / len(cs))
			max_errs[i] = max(err)

		idxes_sort = np.argsort(mse_errs)
		canvas.ax.plot(mse_errs[idxes_sort], c = "orange", label = "mse_err_2nd", zorder = 12)
		ax2.plot(max_errs[idxes_sort], c = "violet", lw = 0.6, label = "max_err_2nd", zorder = 2)
	fw.close()
	canvas.ax.plot(
		[0, corners_count], [mse_err_th, mse_err_th],
		dashes = [12, 5, 4, 5], c = "crimson",
		label = "mse_err_th", lw = 1, zorder = 20,
	)
	ax2.plot(
		[0, corners_count], [max_err_th, max_err_th],
		dashes = [12, 5, 4, 5], c = "silver",
		label = "max_err_th", lw = 1, zorder = 21,
	)
	ax2.tick_params(direction = 'in')
	canvas.ax.legend(loc = "upper left", fontsize = 10, framealpha = 0,
	                 facecolor = None, edgecolor = None)
	ax2.legend(loc = "upper center", fontsize = 10, framealpha = 0,
	           facecolor = None, edgecolor = None)
	canvas.set_axis(equal_axis = False, sci_on = False)

	canvas.save(info_folder + "calib_err.jpg")
	return mat, dist

# def camera_xyz2uv(pos, mat):
# 	"""
#
# 	:param pos:
# 	:param mat:
# 	:return:
# 	"""
# 	pos = np.atleast_2d(pos)
# 	mat = np.asmatrix(np.array(mat))
# 	length = len(pos)
# 	uv = mat * pos.transpose()
# 	uv = np.asarray(uv).transpose()
# 	tmp = np.tile(uv[:, 2], (2, 1))
# 	uv = uv[:, :2] / tmp.transpose()
# 	if length > 1: return uv
# 	return uv.ravel()


# def camera_uv2xyz(uv, mat, z = 1.0):
# 	"""
#
# 	:param uv:
# 	:param mat:
# 	:param z:
# 	:return:
# 	"""
# 	uv = np.atleast_2d(uv)
# 	length = len(uv)
# 	add = np.ones((length, 1))
# 	pos = np.concatenate((uv, add), axis = 1)
# 	pos = np.linalg.inv(mat) * pos.transpose()
# 	pos = np.asarray(pos).transpose() * z
# 	if length > 1: return pos
# 	return pos.ravel()


# def chessboard_draw_pose(src, M_w2c, camera_mat, board_shape, board_r, offset = (0, 0, 0)):
# 	"""
# 	# todo: debug needed
# 	:param src:
# 	:param M_w2c:
# 	:param camera_mat:
# 	:param board_shape:
# 	:param board_r:
# 	:param offset:
# 	:return:
# 	"""
# 	dst = np.uint16(src)
# 	X_w2p = np.asarray([
# 		[0, 0, 0],
# 		[(board_shape[0] - 1) / 2, 0, 0],
# 		[0, (board_shape[1] - 1) / 2, 0],
# 		[0, 0, min(board_shape[0], board_shape[1])],
# 	], np.float64) * board_r + offset
# 	M_c2w = np.linalg.inv(M_w2c)
# 	X_c2p = CoordSys.conv(M_c2w, X_w2p)
# 	uvn = num2int(camera_xyz2uv(X_c2p, camera_mat))
#
# 	colors = KaisColor.plotColor(KaisColor.axis_cnames, gbr = True)
# 	cv2.line(dst, tuple(uvn[0]), tuple(uvn[1]), colors[0], thickness = 3)
# 	cv2.line(dst, tuple(uvn[0]), tuple(uvn[2]), colors[1], thickness = 3)
# 	cv2.drawMarker(dst, tuple(uvn[0]), (255, 255, 255), markerType = cv2.MARKER_CROSS, markerSize = 24, thickness = 3)
# 	cv2.line(dst, tuple(uvn[0]), tuple(uvn[3]), colors[2], thickness = 3)
#
# 	org = tuple(uvn[0] + [30, 90])
# 	color = KaisColor.plotColor("indigo", gbr = True)
# 	info = f"dis = {np.linalg.norm(M_w2c[:3, 3]) / 1000:.4f} m "
# 	cv2.putText(dst, info, org, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (255, 255, 255), 8)
# 	cv2.putText(dst, info, org, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, color, 4)
# 	return dst


# def chessboard_Mat_w2c_webCamera(src, camera_mat, board_shape, board_r):
# 	"""
# 	:param src: src_image
# 	:param camera_mat: 3x3 array
# 	:param board_shape:
# 	:param board_r:
# 	:return: pose matrix world -> camera
# 	"""
#
# 	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# 	imagePoints = chessboard_corner_detect(gray, board_shape)
# 	if imagePoints is None: return None
# 	objectPoints = chessboard_corner_3d(board_shape, board_r)
#
# 	_, rvec, tvec = cv2.solvePnP(
# 		objectPoints, imagePoints, camera_mat, None,
# 		flags = cv2.SOLVEPNP_IPPE
# 	)
# 	R, _ = cv2.Rodrigues(rvec)
# 	M_c2w = np.asmatrix(np.eye(4))
# 	M_c2w[:3, :3] = R
# 	M_c2w[:3, 3] = tvec
# 	# M_c2w *= np.asmatrix("1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1")
# 	return np.linalg.inv(M_c2w)
