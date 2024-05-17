"""
#  process:
#  target image -----> local 3D_cs -----> base 3D_cs -----> project Us
#            resampling <-- +        H                          |
#                           | inv_H                             V
#                           |                              min (u_0, v_0)
#  base image          base 3D_cs <----- project Us <----- min-max (w, h)
#                                                          image Us
"""
import cv2
import numpy as np

from Core.Basic import KaisLog

log = KaisLog.get_log()

def image_ring(w, h, K: int = 4):
	tmp_w = np.linspace(0, w - 1, w // K, dtype = np.float32)
	tmp_h = np.linspace(0, h - 1, h // K, dtype = np.float32)
	tmp_w2 = np.zeros(len(tmp_h), dtype = np.float32)
	tmp_h2 = np.zeros(len(tmp_w), dtype = np.float32)
	u = np.concatenate([tmp_w, tmp_w2, tmp_w, tmp_w2 + w - 1], dtype = np.float32).ravel()
	v = np.concatenate([tmp_h2, tmp_h, tmp_h2 + h - 1, tmp_h], dtype = np.float32).ravel()
	return np.asarray([u, v], np.float32).T


def mask_ring(mask, K: int = 4):
	kernel = np.ones((3, 3), np.uint8)
	K_half = K // 2
	mask = mask[K_half::K, K_half::K]
	sub_mask = cv2.erode(mask, kernel, borderType = cv2.BORDER_CONSTANT, borderValue = 0)
	v, u = np.where(mask - sub_mask > 100)
	return np.asarray([u, v], np.float32).T * K + K_half


def remapping(src, mask_src, roi_src, H, project_func_src, recover_func_src, focal_src,
              project_func_dst, recover_func_dst, focal_dst, *, aspect = 1,
              interpolation = cv2.INTER_LINEAR, borderType = None):
	"""
	memo: H * Us_src -> Us_base
	:param src: could be None
	:param mask_src:
	:param roi_src: (u0, v0, u1, v1), for frame -> (-cx, -cy, w - cx, h - cy)
	:param H: (3, 3) array
	:param project_func_src: func(x, y, z) -> (u, v)
	:param recover_func_src: func(u, v) -> (x, y, z)
	:param focal_src:
	:param project_func_dst: func(x, y, z) -> (u, v)
	:param recover_func_dst: func(u, v) -> (x, y, z)
	:param focal_dst:
	:param aspect: input image's focal_x / focal_y
	:param interpolation:
	:param borderType:
	:return:
		dst:
		mask:
		roi:
	"""
	# warp perspective image ring into base-cs
	if mask_src is None:
		h, w = src.shape[:2]
		U_ring = image_ring(w, h)
		mask_src = np.full((h, w), 255, np.uint8)
	else:
		U_ring = mask_ring(mask_src)
	assert len(U_ring) > 4, ">> error, empty mask .."
	U_ring += (roi_src[0], roi_src[1])
	P_ring = recover_func_src(U_ring, focal_src, aspect)
	if H is not None:
		P_ring = P_ring.dot(H.T)
	U_ring = project_func_dst(P_ring, focal_dst)
	# calc origin roi
	u0 = np.min(U_ring[:, 0])
	v0 = np.min(U_ring[:, 1])
	u1 = np.max(U_ring[:, 0])
	v1 = np.max(U_ring[:, 1])
	new_w = int(u1 - u0 + 0.5)
	new_h = int(v1 - v0 + 0.5)
	log.info(f"new size = ( {new_w}, {new_h} )")
	roi_dst = (u0, v0, u0 + new_w, v0 + new_h)
	# get image idxes
	v_idxes, u_idxes = np.indices((new_h, new_w), int)
	u_idxes = u_idxes.ravel()
	v_idxes = v_idxes.ravel()
	U = np.asarray([u_idxes, v_idxes], np.float32).T
	U += (roi_dst[0] + 0.5, roi_dst[1] + 0.5)
	# recover projection
	P_dst = recover_func_dst(U, focal_dst)
	if H is not None:
		P_dst = P_dst.dot(np.linalg.inv(H).T)
	# locate pts in src image
	invalid = P_dst[:, 2] <= 0
	U_src = project_func_src(P_dst, focal_src, aspect)
	U_src -= roi_src[:2]
	# resampling
	map_u = np.asarray(U_src[:, 0], np.float32).reshape((new_h, new_w))
	map_v = np.asarray(U_src[:, 1], np.float32).reshape((new_h, new_w))
	if src is not None:
		dst = cv2.remap(src, map_u, map_v, interpolation, borderMode = borderType)
	else: dst = None
	mask_dst = cv2.remap(mask_src, map_u, map_v, cv2.INTER_NEAREST)
	mask_dst[v_idxes[invalid], u_idxes[invalid]] = 0
	return dst, mask_dst, roi_dst


def resize(src, mask_src, roi_src, K, interpolation = cv2.INTER_LINEAR):
	dst = cv2.resize(src, None, fx = K, fy = K, interpolation = interpolation)
	mask_dst = cv2.resize(mask_src, None, fx = K, fy = K, interpolation = cv2.INTER_NEAREST)
	h_dst, w_dst = mask_dst.shape
	u0 = roi_src[0] * K
	v0 = roi_src[1] * K
	u1 = u0 + w_dst
	v1 = v0 + h_dst
	roi_dst = (u0, v0, u1, v1)
	return dst, mask_dst, roi_dst


if __name__ == '__main__':
	def test_remapping():
		from Core.ComputerVision.Replaceable.projection import project_cylinder, recover_cylinder, \
			project_flat, recover_flat, project_stereographic, recover_stereographic
		src = cv2.imread("/home/kai/Desktop/DailySource/M23_03/D2303_22/source/IMG_3205.jpeg")
		focal = 2000
		h, w = src.shape[:2]
		roi_src = (-w // 2, -h // 2, w // 2, h // 2)
		mask_src = np.full((h, w), 255, np.uint8)
		dst, mask, roi = remapping(src, mask_src, roi_src, focal, None,
		                           project_stereographic, recover_stereographic,
		                           project_cylinder, recover_cylinder)
		print(roi)
		cv2.imwrite("/home/kai/Desktop/DailySource/M23_03/D2303_23/out/out.jpg", dst)
		cv2.imwrite("/home/kai/Desktop/DailySource/M23_03/D2303_23/out/mask.jpg", mask)
		# print(offset)
		pass


	test_remapping()
	pass
