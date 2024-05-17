if __name__ == '__main__':
	import numpy as np

	from Core.CodeQuality import KaisTest


	# mark: best
	def test_1(U, f, c):
		u = (U[:, 0] - c) / f
		v = (U[:, 1] - c) / f
		return np.asarray([u, v], U.dtype).T


	def test_1x(U, f, c):
		u = U[:, 0] / f - c / f
		v = U[:, 1] / f - c / f
		return np.asarray([u, v], U.dtype).T


	def test_2(U, f, c):
		U_ = np.zeros_like(U)
		U_[:, 0] = (U[:, 0] - c) / f
		U_[:, 1] = (U[:, 1] - c) / f
		return U_


	def test_3(U, f, c):
		U_ = (U - (c, c)) / np.tile((f, f), (len(U), 1))
		return U_


	def main():
		U = np.random.random((4000000, 2)) * 1000
		f = 500
		c = 500
		KaisTest.func_speed_test(
			members = [
				KaisTest.TestMember(
					func = test_1,
					args = (U, f, c),
					name = "func 01",
				),
				KaisTest.TestMember(
					func = test_1x,
					args = (U, f, c),
					name = "func 01x",
				),
				KaisTest.TestMember(
					func = test_2,
					args = (U, f, c),
					name = "func 02",
				),
				KaisTest.TestMember(
					func = test_3,
					args = (U, f, c),
					name = "func 03",
				),
			],
			repeat_num = 10,
			loop_num = 5,
		)

		pass


	main()
	pass
