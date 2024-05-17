import cv2

if __name__ == '__main__':
	import numpy as np

	from Core.CodeQuality import KaisTest

	folder_o = "/home/kai/Desktop/DailySource/M23_04/D2304_01/out/"
	img = cv2.imread("/home/kai/Desktop/DailySource/M23_03/D2303_20/source/OctopathTravelerMap_v3_ultra.png")


	# mark: best for resize
	def test_1(K):
		tmp = cv2.resize(img, None, fx = K, fy = K, interpolation = cv2.INTER_CUBIC)
		# cv2.imwrite(folder_o + "func_1.png", tmp)
		return tmp


	def test_2(K):
		h_src, w_src = img.shape[:2]
		w_dst = int(w_src * K + 0.5)
		h_dst = int(h_src * K + 0.5)
		v, u = np.mgrid[:h_dst, :w_dst]  # mark: do not use mgrid any more ..
		u = np.float32(u / K + 0.5)
		v = np.float32(v / K + 0.5)
		tmp = cv2.remap(img, u, v, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
		# cv2.imwrite(folder_o + "func_2.png", tmp)
		return tmp


	# mark: 2nd best, if have to use remap
	def test_3(K):
		h_src, w_src = img.shape[:2]
		w_dst = int(w_src * K + 0.5)
		h_dst = int(h_src * K + 0.5)
		v, u = np.indices((h_dst, w_dst), np.float32)
		u = u / K + 0.5
		v = v / K + 0.5
		tmp = cv2.remap(img, u, v, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
		# cv2.imwrite(folder_o + "func_3.png", tmp)
		return tmp


	def test_4(K):
		h_src, w_src = img.shape[:2]
		w_dst = int(w_src * K + 0.5)
		h_dst = int(h_src * K + 0.5)
		v, u = np.indices((h_dst, w_dst), np.float32)
		u = u / K + 0.5
		v = v / K + 0.5
		# tmp = cv2.remap(img, u, v, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
		# cv2.imwrite(folder_o + "func_3.png", tmp)
		# mark: 对照组
		tmp = cv2.resize(img, None, fx = K, fy = K, interpolation = cv2.INTER_CUBIC)
		return tmp


	def main():
		K = 6 / 7
		KaisTest.func_speed_test(
			members = [
				KaisTest.TestMember(
					func = test_1,
					args = (K,),
					name = "func 01",
				),
				KaisTest.TestMember(
					func = test_4,
					args = (K,),
					name = "func 04",
				),
				KaisTest.TestMember(
					func = test_3,
					args = (K,),
					name = "func 03",
				),
			],
			repeat_num = 10,
			loop_num = 5,
		)
		pass


	main()
	# t1 = test_1(0.5)
	# t2 = test_2(0.5)
	# diff = cv2.absdiff(t1, t2)
	# print(diff.shape)
	# cv2.imwrite(folder_o + "diff.png", diff)
	pass
