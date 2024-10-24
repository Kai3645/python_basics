import time

import numpy as np


class KaisTest:
	class TestMember:
		def __init__(self, func, args: tuple = None, para: dict = None, name: str = ""):
			self.name = name
			self.func = func
			if args is None: args = ()
			self.args = args
			if para is None: para = {}
			self.kwargs = para
		
		def run(self, repeat: int):
			t0 = time.time()
			for i in range(repeat):
				self.func(*self.args, **self.kwargs)
			dt = (time.time() - t0) * 1000
			print(f">> {self.name} -> dt = {dt:.1f} ms")
			return dt
	
	@staticmethod
	def func_speed_test(members, repeat_num: int, loop_num: int = 1):
		print(">> warming up PC .. ")
		print("- - - - - - - - - -")
		warm_repeat = max(1, int(repeat_num * 0.1))
		for i, m in enumerate(members): m.run(warm_repeat)
		for i, m in enumerate(members): m.run(warm_repeat)
		print("===========================================\n")
		
		print(">> test start .. ")
		data = np.zeros(len(members))
		for loop_count in range(loop_num):
			print("- - - - - - - - - -")
			print(f"loop: {loop_count + 1}")
			for i, m in enumerate(members): data[i] += m.run(repeat_num)
		print("===========================================\n")
		
		print(">> result")
		data /= loop_num
		for i in np.argsort(data):
			print(f"{members[i].name} -> time costs {data[i]:.3f} ms /{repeat_num} times")
		print("===========================================\n")
		pass
	
	# todo: add other tests
	
	pass


if __name__ == "__main__":
	import random
	
	
	def f1(arr):
		return random.sample(list(arr), len(arr))
	
	
	def f2(arr):
		return np.random.choice(arr, len(arr), False)
	
	
	def f3(arr):
		src = np.asanyarray(arr)
		idxes = np.argsort(np.random.random(len(src)))
		return src[idxes]
	
	
	def main():
		array = [round(random.random(), 3) for _ in range(10000)]
		
		KaisTest.func_speed_test(
			members = [
				KaisTest.TestMember(
					func = f1,
					args = (array,),
					name = "func 01",
				),
				KaisTest.TestMember(
					func = f2,
					args = (array,),
					name = "func 02",
				),
				KaisTest.TestMember(
					func = f3,
					args = (array,),
					name = "func 03",
				),
			],
			repeat_num = 30,
			loop_num = 5,
		)
	
	
	main()
	pass
