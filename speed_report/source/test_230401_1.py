import cv2

from Core.ComputerVision.Replaceable import remapping, project_cylinder, recover_flat, project_flat, recover_cylinder
from Core.ComputerVision.basic_image import mask_edge_alpha_fast, mask_edge_alpha

if __name__ == '__main__':
	import numpy as np

	from Core.CodeQuality import KaisTest

	folder_o = "/home/kai/Desktop/DailySource/M23_04/D2304_01/out/"

	W = 5000
	H = 3000
	CX = W / 2
	CY = H / 2
	mask = np.full((H, W), 255, np.uint8)
	roi = (-CX, -CY, CX, CY)
	focal = W * 0.8
	H = np.asarray([[0.75, 0, 0], [0, 0.75, 0], [0, 0, 1]])
	_, mask, roi_i = remapping(None, mask, roi, focal, H, project_flat, recover_flat,
	                           project_cylinder, recover_cylinder)



	def test_1():
		tmp = mask_edge_alpha(mask, 200)
		tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(folder_o + "func_1.png", tmp)
		return


	def test_2():
		tmp = mask_edge_alpha_fast(mask, 200, 1)
		tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(folder_o + "func_2.png", tmp)
		return


	# mark: best, consider result
	def test_3():
		tmp = mask_edge_alpha_fast(mask, 200, 3)
		tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(folder_o + "func_3.png", tmp)
		return


	def test_4():
		tmp = mask_edge_alpha_fast(mask, 200, 5)
		tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(folder_o + "func_4.png", tmp)
		return


	def main():
		KaisTest.func_speed_test(
			members = [
				KaisTest.TestMember(
					func = test_1,
					args = None,
					name = "func 01",
				),
				KaisTest.TestMember(
					func = test_2,
					args = None,
					name = "func 02",
				),
				KaisTest.TestMember(
					func = test_3,
					args = None,
					name = "func 03",
				),
				KaisTest.TestMember(
					func = test_4,
					args = None,
					name = "func 04",
				),
			],
			repeat_num = 10,
			loop_num = 5,
		)
		pass


	main()
	# test_1()
	# test_2()
	# test_3()
	# test_4()
	pass
