import time

import numpy as np


class Sudoku:
	def __init__(self, board, xy2uv):
		"""
		x, i -> row
		y, j -> column
		u, m -> block
		v, n -> block local index
		d -> depth
		type id:
			0->row
			1->column
			2->block
			3->depth
		:param board: [0, 9], 0 -> empty
		:param xy2uv:
		"""

		class Node:
			def __init__(self):
				self.info = None  # row, column, depth
				self.H_ids = None  # head indexes
				pass

		# 外显数据
		self.result = board
		self.saved_results = []  # 解的存储

		# 检索表
		self.candies = [Node() for _ in range(729)]
		self.heads = [set() for _ in range(324)]
		self.scanners = [set(range(0, 81)), set(range(81, 162)), set(range(162, 243)), set(range(243, 324))]

		self.compliance_flag = True
		self.step_recorder = [0] * 729
		self.step_id = 0
		self.guess_list = []

		self.xy2uv = xy2uv
		# 初始候选表
		for i in range(9):
			for j in range(9):
				for d in range(9):
					u = self.xy2uv[i][j][0]
					item_id = i * 81 + j * 9 + d
					H0 = i * 9 + d
					H1 = j * 9 + d + 81
					H2 = u * 9 + d + 162
					H3 = i * 9 + j + 243

					self.candies[item_id].info = i, j, d
					self.candies[item_id].H_ids = H0, H1, H2, H3

					self.heads[H0].add(item_id)
					self.heads[H1].add(item_id)
					self.heads[H2].add(item_id)
					self.heads[H3].add(item_id)

		# 初始数值
		for i in range(9):
			for j in range(9):
				if board[i][j] == 0: continue
				d = board[i][j] - 1
				item_id = i * 81 + j * 9 + d
				self.hide_candy(item_id)
				self.prune_links(item_id)
		pass

	def check_unique(self):
		for scanner in self.scanners:
			for H_id in scanner:
				length = len(self.heads[H_id])
				if length == 0:
					self.compliance_flag = False
					return -1
				elif length == 1:
					item_id = next(iter(self.heads[H_id]))
					for H_id_ in self.candies[item_id].H_ids:
						if len(self.heads[H_id_]) == 0:
							self.compliance_flag = False
							return -1
					return item_id
		return -1

	def hide_candy(self, item_id):
		self.step_recorder[self.step_id] = item_id
		self.step_id += 1
		for H_id in self.candies[item_id].H_ids:
			self.heads[H_id].discard(item_id)
		pass

	def prune_links(self, item_id):
		for t, H_id in enumerate(self.candies[item_id].H_ids):
			for hook in self.heads[H_id].copy():
				self.hide_candy(hook)
			self.scanners[t].discard(H_id)

	def show_candy(self, item_id):
		for t, H_id in enumerate(self.candies[item_id].H_ids):
			if len(self.heads[H_id]) == 0:
				self.scanners[t].add(H_id)
			self.heads[H_id].add(item_id)
		pass

	def get_optimal_candy(self, t = 3):
		best_size = 10
		best_id = -1
		for H_id in self.scanners[t]:
			size = len(self.heads[H_id])
			if size == 2: return next(iter(self.heads[H_id]))  # shortcut
			elif size < best_size:
				best_size = size
				best_id = next(iter(self.heads[H_id]))
		return best_id

	def upgrade(self):
		while self.compliance_flag:
			item_id = self.check_unique()
			if item_id == -1: return self.compliance_flag
			x, y, d = self.candies[item_id].info
			self.result[x][y] = d + 1
			self.hide_candy(item_id)
			self.prune_links(item_id)
		return False

	def step_forward(self):
		item_id = self.get_optimal_candy(3)
		x, y, d = self.candies[item_id].info
		self.result[x][y] = d + 1
		self.guess_list.append(self.step_id)
		print(f"{self.guess_list} -> ({x + 1}, {y + 1}) = {d + 1}")
		self.hide_candy(item_id)
		self.prune_links(item_id)

	def step_back(self):
		if len(self.guess_list) == 0: return False
		self.compliance_flag = True
		end_id = self.guess_list.pop()
		start_id = self.step_id - 1
		for si in range(start_id, end_id, -1):
			item_id = self.step_recorder[si]
			self.show_candy(item_id)
		self.step_id = end_id + 1
		return True

	def solve(self, max_result):
		count = 1
		flag = True
		while flag:
			if not self.upgrade():
				flag = self.step_back()
				continue
			if len(self.scanners[3]) > 0:
				self.step_forward()
				continue
			print(f"-> got result {count:02d}")
			self.saved_results.append(np.asarray(self.result))
			if count > max_result:
				break
			count += 1
			flag = self.step_back()
		pass

	def __str__(self):
		tmp = "╔═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╗\n"
		for i in range(9):
			if i > 0:
				tmp += "║"
				for j in range(9):
					if j > 0:
						tmp += "┼"
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i - 1][j][0]
					if m1 == m2:
						tmp += "∙∙∙"
					else:
						tmp += "═══"
				tmp += "║\n"
			tmp += "║ "
			for j in range(9):
				if j > 0:
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i][j - 1][0]
					if m1 == m2:
						tmp += " : "
					else:
						tmp += " ║ "
				if self.result[i][j] > 0:
					tmp += f"{self.result[i][j]}"
				else:
					tmp += " "
				pass
			tmp += " ║\n"
		tmp += "╚═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╝"
		return tmp


if __name__ == '__main__':
	"""
	 8 . . | . . . | . . .
	 . . 3 | 6 . . | . . .
	 . 7 . | . 9 . | 2 . .
	 ------+-------+------ 
	 . 5 . | . . 7 | . . .
	 . . . | . 4 5 | 7 . .
	 . . . | 1 . . | . 3 .
	 ------+-------+------ 
	 . . 1 | . . . | . 6 8
	 . . 8 | 5 . . | . 1 .
	 . 9 . | . . . | 4 . .
	"""


	def main():
		max_result = 16
		t0 = time.time()

		xy2uv = [[[0, 0] for _ in range(9)] for _ in range(9)]
		uv2xy = [[[0, 0] for _ in range(9)] for _ in range(9)]
		block = [
			[1, 1, 1, 2, 2, 2, 3, 3, 3, ],
			[1, 1, 1, 2, 2, 2, 3, 3, 3, ],
			[1, 1, 1, 2, 2, 2, 3, 3, 3, ],
			[4, 4, 4, 5, 5, 5, 6, 6, 6, ],
			[4, 4, 4, 5, 5, 5, 6, 6, 6, ],
			[4, 4, 4, 5, 5, 5, 6, 6, 6, ],
			[7, 7, 7, 8, 8, 8, 9, 9, 9, ],
			[7, 7, 7, 8, 8, 8, 9, 9, 9, ],
			[7, 7, 7, 8, 8, 8, 9, 9, 9, ],
		]
		for m in range(9):
			n = 0
			for i in range(9):
				for j in range(9):
					if block[i][j] - 1 != m: continue
					xy2uv[i][j][:] = m, n
					uv2xy[m][n][:] = i, j
					n += 1
			pass

		sudoku = Sudoku(board = [
			[8, 0, 0, 0, 0, 0, 0, 0, 0, ],
			[0, 0, 3, 6, 0, 0, 0, 0, 0, ],
			[0, 7, 0, 0, 9, 0, 2, 0, 0, ],
			[0, 5, 0, 0, 0, 7, 0, 0, 0, ],
			[0, 0, 0, 0, 4, 5, 7, 0, 0, ],
			[0, 0, 0, 1, 0, 0, 0, 3, 0, ],
			[0, 0, 1, 0, 0, 0, 0, 6, 8, ],
			[0, 0, 8, 5, 0, 0, 0, 1, 0, ],
			[0, 9, 0, 0, 0, 0, 4, 0, 0, ],
		], xy2uv = xy2uv)
		print(f"init time = {time.time() - t0:.4f} s")
		print(sudoku)

		t1 = time.time()
		sudoku.solve(max_result)
		print(f"solve time = {time.time() - t1:.4f} s")

		print(sudoku.saved_results)


	main()
	pass
