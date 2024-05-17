if __name__ == '__main__':
	import numpy as np

	from Core.CodeQuality import KaisTest

	H = 10000
	W = 15000


	# mark: best, same to [2] & [4], but easy for user
	def test_1():
		v, u = np.indices((H, W), np.float32)
		return v, u


	def test_2():
		v = np.empty((H, W), np.float32)
		idx = np.arange(H, dtype = np.float32).reshape((H, 1))
		v[:] = idx
		u = np.empty((H, W), np.float32)
		idx = np.arange(W, dtype = np.float32).reshape((1, W))
		u[:] = idx
		return v, u


	def test_3():
		# mark: data type changing costs a lot of time
		v = np.tile(range(H), (W, 1)).T.astype(np.float32)
		u = np.tile(range(W), (H, 1)).astype(np.float32)
		return v, u


	def test_4():
		v = np.tile(np.arange(H, dtype = np.float32), (W, 1)).T
		u = np.tile(np.arange(W, dtype = np.float32), (H, 1))
		return v, u


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
	pass
