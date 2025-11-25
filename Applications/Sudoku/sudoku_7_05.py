import os
import sys
import time

import numpy as np


class Sudoku:
	def __init__(self):
		"""
		x, i -> row
		y, j -> column
		u, m -> block
		v, n -> block local index
		d -> depth
		t -> type:
			0 -> row    x depth
			1 -> column x depth
			2 -> block  x depth
			3 -> row    x column
		"""
		# 外显数据
		self.result = None
		self.saved_results = []  # 解的存储

		# 检索表
		self.nodes_flag = [True] * 729
		self.nodes_info = [(0, 0, 0) for _ in range(729)]
		self.nodes_Hs = [(0, 0, 0, 0) for _ in range(729)]
		self.heads = [set() for _ in range(324)]
		self.scanners = [set(range(0, 81)), set(range(81, 162)), set(range(162, 243)), set(range(243, 324))]
		self.compliance_flag = True

		self.step_recorder = [0] * 729
		self.step_id = 0
		self.guess_list = []

		self.xy2uv = [[[0, 0] for _ in range(9)] for _ in range(9)]
		self.uv2xy = [[[0, 0] for _ in range(9)] for _ in range(9)]

		self.logs = []
		pass

	def setup(self, board, block):
		# 外显数据
		t0 = time.time()
		self.result = board

		for m in range(9):
			n = 0
			for i in range(9):
				for j in range(9):
					if block[i][j] - 1 != m: continue
					self.xy2uv[i][j][0] = m
					self.xy2uv[i][j][1] = n
					self.uv2xy[m][n][0] = i
					self.uv2xy[m][n][1] = j
					n += 1

		tmp_str = ">> block data\n"
		tmp_str += self.str_data(block)
		self.logs.append(tmp_str)
		print(tmp_str)

		tmp_str = ">> board data\n"
		tmp_str += self.str_data(board)
		self.logs.append(tmp_str)
		print(tmp_str)

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

					self.nodes_info[item_id] = i, j, d
					self.nodes_Hs[item_id] = H0, H1, H2, H3

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

		tmp_str = f"initiate time = {time.time() - t0:.4f} s\n"
		self.logs.append(tmp_str)
		print(tmp_str)

	def check_unique(self):
		for scanner in self.scanners:
			for H_id in scanner:
				length = len(self.heads[H_id])
				if length == 0:
					self.compliance_flag = False
					return -1
				elif length == 1:
					item_id = next(iter(self.heads[H_id]))
					for H_id_ in self.nodes_Hs[item_id]:
						if len(self.heads[H_id_]) == 0:
							self.compliance_flag = False
							return -1
					return item_id
		return -1

	def hide_candy(self, item_id):
		if not self.nodes_flag[item_id]: return
		self.nodes_flag[item_id] = False
		self.step_recorder[self.step_id] = item_id
		self.step_id += 1
		for H_id in self.nodes_Hs[item_id]:
			self.heads[H_id].remove(item_id)
		pass

	def prune_links(self, item_id):
		for t, H_id in enumerate(self.nodes_Hs[item_id]):
			for hook in self.heads[H_id].copy():
				self.hide_candy(hook)
			self.scanners[t].discard(H_id)

	def show_candy(self, item_id):
		for t, H_id in enumerate(self.nodes_Hs[item_id]):
			if len(self.heads[H_id]) == 0:
				self.scanners[t].add(H_id)
			self.heads[H_id].add(item_id)
		self.nodes_flag[item_id] = True
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
			x, y, d = self.nodes_info[item_id]
			self.result[x][y] = d + 1
			self.hide_candy(item_id)
			self.prune_links(item_id)
		return False

	def step_forward(self):
		item_id = self.get_optimal_candy(3)
		x, y, d = self.nodes_info[item_id]
		self.result[x][y] = d + 1
		self.guess_list.append(self.step_id)
		tmp_str = "... " * (len(self.guess_list) - 1)
		tmp_str += f"{self.guess_list[-1]:03d} --> ({x + 1}, {y + 1}) = {d + 1}"
		self.logs.append(tmp_str)
		print(tmp_str)
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
		t0 = time.time()
		tmp_str = ">> start"
		self.logs.append(tmp_str)
		print(tmp_str)
		count = 0
		flag = True
		is_first = True
		while flag:
			if not self.upgrade():
				flag = self.step_back()
				continue
			if len(self.scanners[3]) > 0:
				if is_first:
					tmp_str = ">> candies\n"
					tmp_str += self.str_candies()
					self.logs.append(tmp_str)
					print(tmp_str)
					is_first = False
				self.step_forward()
				continue
			count += 1
			tmp_str = f">> result {count:02d}\n"
			tmp_str += self.str_result()
			self.logs.append(tmp_str)
			print(tmp_str)
			self.saved_results.append(np.asarray(self.result))
			if count >= max_result:
				tmp_str = f">> exit, total result over {max_result}"
				self.logs.append(tmp_str)
				print(tmp_str)
				break
			flag = self.step_back()
		tmp_str = ">> finished\n"
		tmp_str += f"solving time = {time.time() - t0:.4f} s\n"
		if count == 0: tmp_str += "no result detected\n"
		self.logs.append(tmp_str)
		print(tmp_str)

	def str_candies(self):
		code2str = [
			"ˍˍˍˍˍˍˍˍˍ", "1ˍˍˍˍˍˍˍˍ", "ˍ2ˍˍˍˍˍˍˍ", "12ˍˍˍˍˍˍˍ", "ˍˍ3ˍˍˍˍˍˍ", "1ˍ3ˍˍˍˍˍˍ", "ˍ23ˍˍˍˍˍˍ", "123ˍˍˍˍˍˍ",
			"ˍˍˍ4ˍˍˍˍˍ", "1ˍˍ4ˍˍˍˍˍ", "ˍ2ˍ4ˍˍˍˍˍ", "12ˍ4ˍˍˍˍˍ", "ˍˍ34ˍˍˍˍˍ", "1ˍ34ˍˍˍˍˍ", "ˍ234ˍˍˍˍˍ", "1234ˍˍˍˍˍ",
			"ˍˍˍˍ5ˍˍˍˍ", "1ˍˍˍ5ˍˍˍˍ", "ˍ2ˍˍ5ˍˍˍˍ", "12ˍˍ5ˍˍˍˍ", "ˍˍ3ˍ5ˍˍˍˍ", "1ˍ3ˍ5ˍˍˍˍ", "ˍ23ˍ5ˍˍˍˍ", "123ˍ5ˍˍˍˍ",
			"ˍˍˍ45ˍˍˍˍ", "1ˍˍ45ˍˍˍˍ", "ˍ2ˍ45ˍˍˍˍ", "12ˍ45ˍˍˍˍ", "ˍˍ345ˍˍˍˍ", "1ˍ345ˍˍˍˍ", "ˍ2345ˍˍˍˍ", "12345ˍˍˍˍ",
			"ˍˍˍˍˍ6ˍˍˍ", "1ˍˍˍˍ6ˍˍˍ", "ˍ2ˍˍˍ6ˍˍˍ", "12ˍˍˍ6ˍˍˍ", "ˍˍ3ˍˍ6ˍˍˍ", "1ˍ3ˍˍ6ˍˍˍ", "ˍ23ˍˍ6ˍˍˍ", "123ˍˍ6ˍˍˍ",
			"ˍˍˍ4ˍ6ˍˍˍ", "1ˍˍ4ˍ6ˍˍˍ", "ˍ2ˍ4ˍ6ˍˍˍ", "12ˍ4ˍ6ˍˍˍ", "ˍˍ34ˍ6ˍˍˍ", "1ˍ34ˍ6ˍˍˍ", "ˍ234ˍ6ˍˍˍ", "1234ˍ6ˍˍˍ",
			"ˍˍˍˍ56ˍˍˍ", "1ˍˍˍ56ˍˍˍ", "ˍ2ˍˍ56ˍˍˍ", "12ˍˍ56ˍˍˍ", "ˍˍ3ˍ56ˍˍˍ", "1ˍ3ˍ56ˍˍˍ", "ˍ23ˍ56ˍˍˍ", "123ˍ56ˍˍˍ",
			"ˍˍˍ456ˍˍˍ", "1ˍˍ456ˍˍˍ", "ˍ2ˍ456ˍˍˍ", "12ˍ456ˍˍˍ", "ˍˍ3456ˍˍˍ", "1ˍ3456ˍˍˍ", "ˍ23456ˍˍˍ", "123456ˍˍˍ",
			"ˍˍˍˍˍˍ7ˍˍ", "1ˍˍˍˍˍ7ˍˍ", "ˍ2ˍˍˍˍ7ˍˍ", "12ˍˍˍˍ7ˍˍ", "ˍˍ3ˍˍˍ7ˍˍ", "1ˍ3ˍˍˍ7ˍˍ", "ˍ23ˍˍˍ7ˍˍ", "123ˍˍˍ7ˍˍ",
			"ˍˍˍ4ˍˍ7ˍˍ", "1ˍˍ4ˍˍ7ˍˍ", "ˍ2ˍ4ˍˍ7ˍˍ", "12ˍ4ˍˍ7ˍˍ", "ˍˍ34ˍˍ7ˍˍ", "1ˍ34ˍˍ7ˍˍ", "ˍ234ˍˍ7ˍˍ", "1234ˍˍ7ˍˍ",
			"ˍˍˍˍ5ˍ7ˍˍ", "1ˍˍˍ5ˍ7ˍˍ", "ˍ2ˍˍ5ˍ7ˍˍ", "12ˍˍ5ˍ7ˍˍ", "ˍˍ3ˍ5ˍ7ˍˍ", "1ˍ3ˍ5ˍ7ˍˍ", "ˍ23ˍ5ˍ7ˍˍ", "123ˍ5ˍ7ˍˍ",
			"ˍˍˍ45ˍ7ˍˍ", "1ˍˍ45ˍ7ˍˍ", "ˍ2ˍ45ˍ7ˍˍ", "12ˍ45ˍ7ˍˍ", "ˍˍ345ˍ7ˍˍ", "1ˍ345ˍ7ˍˍ", "ˍ2345ˍ7ˍˍ", "12345ˍ7ˍˍ",
			"ˍˍˍˍˍ67ˍˍ", "1ˍˍˍˍ67ˍˍ", "ˍ2ˍˍˍ67ˍˍ", "12ˍˍˍ67ˍˍ", "ˍˍ3ˍˍ67ˍˍ", "1ˍ3ˍˍ67ˍˍ", "ˍ23ˍˍ67ˍˍ", "123ˍˍ67ˍˍ",
			"ˍˍˍ4ˍ67ˍˍ", "1ˍˍ4ˍ67ˍˍ", "ˍ2ˍ4ˍ67ˍˍ", "12ˍ4ˍ67ˍˍ", "ˍˍ34ˍ67ˍˍ", "1ˍ34ˍ67ˍˍ", "ˍ234ˍ67ˍˍ", "1234ˍ67ˍˍ",
			"ˍˍˍˍ567ˍˍ", "1ˍˍˍ567ˍˍ", "ˍ2ˍˍ567ˍˍ", "12ˍˍ567ˍˍ", "ˍˍ3ˍ567ˍˍ", "1ˍ3ˍ567ˍˍ", "ˍ23ˍ567ˍˍ", "123ˍ567ˍˍ",
			"ˍˍˍ4567ˍˍ", "1ˍˍ4567ˍˍ", "ˍ2ˍ4567ˍˍ", "12ˍ4567ˍˍ", "ˍˍ34567ˍˍ", "1ˍ34567ˍˍ", "ˍ234567ˍˍ", "1234567ˍˍ",
			"ˍˍˍˍˍˍˍ8ˍ", "1ˍˍˍˍˍˍ8ˍ", "ˍ2ˍˍˍˍˍ8ˍ", "12ˍˍˍˍˍ8ˍ", "ˍˍ3ˍˍˍˍ8ˍ", "1ˍ3ˍˍˍˍ8ˍ", "ˍ23ˍˍˍˍ8ˍ", "123ˍˍˍˍ8ˍ",
			"ˍˍˍ4ˍˍˍ8ˍ", "1ˍˍ4ˍˍˍ8ˍ", "ˍ2ˍ4ˍˍˍ8ˍ", "12ˍ4ˍˍˍ8ˍ", "ˍˍ34ˍˍˍ8ˍ", "1ˍ34ˍˍˍ8ˍ", "ˍ234ˍˍˍ8ˍ", "1234ˍˍˍ8ˍ",
			"ˍˍˍˍ5ˍˍ8ˍ", "1ˍˍˍ5ˍˍ8ˍ", "ˍ2ˍˍ5ˍˍ8ˍ", "12ˍˍ5ˍˍ8ˍ", "ˍˍ3ˍ5ˍˍ8ˍ", "1ˍ3ˍ5ˍˍ8ˍ", "ˍ23ˍ5ˍˍ8ˍ", "123ˍ5ˍˍ8ˍ",
			"ˍˍˍ45ˍˍ8ˍ", "1ˍˍ45ˍˍ8ˍ", "ˍ2ˍ45ˍˍ8ˍ", "12ˍ45ˍˍ8ˍ", "ˍˍ345ˍˍ8ˍ", "1ˍ345ˍˍ8ˍ", "ˍ2345ˍˍ8ˍ", "12345ˍˍ8ˍ",
			"ˍˍˍˍˍ6ˍ8ˍ", "1ˍˍˍˍ6ˍ8ˍ", "ˍ2ˍˍˍ6ˍ8ˍ", "12ˍˍˍ6ˍ8ˍ", "ˍˍ3ˍˍ6ˍ8ˍ", "1ˍ3ˍˍ6ˍ8ˍ", "ˍ23ˍˍ6ˍ8ˍ", "123ˍˍ6ˍ8ˍ",
			"ˍˍˍ4ˍ6ˍ8ˍ", "1ˍˍ4ˍ6ˍ8ˍ", "ˍ2ˍ4ˍ6ˍ8ˍ", "12ˍ4ˍ6ˍ8ˍ", "ˍˍ34ˍ6ˍ8ˍ", "1ˍ34ˍ6ˍ8ˍ", "ˍ234ˍ6ˍ8ˍ", "1234ˍ6ˍ8ˍ",
			"ˍˍˍˍ56ˍ8ˍ", "1ˍˍˍ56ˍ8ˍ", "ˍ2ˍˍ56ˍ8ˍ", "12ˍˍ56ˍ8ˍ", "ˍˍ3ˍ56ˍ8ˍ", "1ˍ3ˍ56ˍ8ˍ", "ˍ23ˍ56ˍ8ˍ", "123ˍ56ˍ8ˍ",
			"ˍˍˍ456ˍ8ˍ", "1ˍˍ456ˍ8ˍ", "ˍ2ˍ456ˍ8ˍ", "12ˍ456ˍ8ˍ", "ˍˍ3456ˍ8ˍ", "1ˍ3456ˍ8ˍ", "ˍ23456ˍ8ˍ", "123456ˍ8ˍ",
			"ˍˍˍˍˍˍ78ˍ", "1ˍˍˍˍˍ78ˍ", "ˍ2ˍˍˍˍ78ˍ", "12ˍˍˍˍ78ˍ", "ˍˍ3ˍˍˍ78ˍ", "1ˍ3ˍˍˍ78ˍ", "ˍ23ˍˍˍ78ˍ", "123ˍˍˍ78ˍ",
			"ˍˍˍ4ˍˍ78ˍ", "1ˍˍ4ˍˍ78ˍ", "ˍ2ˍ4ˍˍ78ˍ", "12ˍ4ˍˍ78ˍ", "ˍˍ34ˍˍ78ˍ", "1ˍ34ˍˍ78ˍ", "ˍ234ˍˍ78ˍ", "1234ˍˍ78ˍ",
			"ˍˍˍˍ5ˍ78ˍ", "1ˍˍˍ5ˍ78ˍ", "ˍ2ˍˍ5ˍ78ˍ", "12ˍˍ5ˍ78ˍ", "ˍˍ3ˍ5ˍ78ˍ", "1ˍ3ˍ5ˍ78ˍ", "ˍ23ˍ5ˍ78ˍ", "123ˍ5ˍ78ˍ",
			"ˍˍˍ45ˍ78ˍ", "1ˍˍ45ˍ78ˍ", "ˍ2ˍ45ˍ78ˍ", "12ˍ45ˍ78ˍ", "ˍˍ345ˍ78ˍ", "1ˍ345ˍ78ˍ", "ˍ2345ˍ78ˍ", "12345ˍ78ˍ",
			"ˍˍˍˍˍ678ˍ", "1ˍˍˍˍ678ˍ", "ˍ2ˍˍˍ678ˍ", "12ˍˍˍ678ˍ", "ˍˍ3ˍˍ678ˍ", "1ˍ3ˍˍ678ˍ", "ˍ23ˍˍ678ˍ", "123ˍˍ678ˍ",
			"ˍˍˍ4ˍ678ˍ", "1ˍˍ4ˍ678ˍ", "ˍ2ˍ4ˍ678ˍ", "12ˍ4ˍ678ˍ", "ˍˍ34ˍ678ˍ", "1ˍ34ˍ678ˍ", "ˍ234ˍ678ˍ", "1234ˍ678ˍ",
			"ˍˍˍˍ5678ˍ", "1ˍˍˍ5678ˍ", "ˍ2ˍˍ5678ˍ", "12ˍˍ5678ˍ", "ˍˍ3ˍ5678ˍ", "1ˍ3ˍ5678ˍ", "ˍ23ˍ5678ˍ", "123ˍ5678ˍ",
			"ˍˍˍ45678ˍ", "1ˍˍ45678ˍ", "ˍ2ˍ45678ˍ", "12ˍ45678ˍ", "ˍˍ345678ˍ", "1ˍ345678ˍ", "ˍ2345678ˍ", "12345678ˍ",
			"ˍˍˍˍˍˍˍˍ9", "1ˍˍˍˍˍˍˍ9", "ˍ2ˍˍˍˍˍˍ9", "12ˍˍˍˍˍˍ9", "ˍˍ3ˍˍˍˍˍ9", "1ˍ3ˍˍˍˍˍ9", "ˍ23ˍˍˍˍˍ9", "123ˍˍˍˍˍ9",
			"ˍˍˍ4ˍˍˍˍ9", "1ˍˍ4ˍˍˍˍ9", "ˍ2ˍ4ˍˍˍˍ9", "12ˍ4ˍˍˍˍ9", "ˍˍ34ˍˍˍˍ9", "1ˍ34ˍˍˍˍ9", "ˍ234ˍˍˍˍ9", "1234ˍˍˍˍ9",
			"ˍˍˍˍ5ˍˍˍ9", "1ˍˍˍ5ˍˍˍ9", "ˍ2ˍˍ5ˍˍˍ9", "12ˍˍ5ˍˍˍ9", "ˍˍ3ˍ5ˍˍˍ9", "1ˍ3ˍ5ˍˍˍ9", "ˍ23ˍ5ˍˍˍ9", "123ˍ5ˍˍˍ9",
			"ˍˍˍ45ˍˍˍ9", "1ˍˍ45ˍˍˍ9", "ˍ2ˍ45ˍˍˍ9", "12ˍ45ˍˍˍ9", "ˍˍ345ˍˍˍ9", "1ˍ345ˍˍˍ9", "ˍ2345ˍˍˍ9", "12345ˍˍˍ9",
			"ˍˍˍˍˍ6ˍˍ9", "1ˍˍˍˍ6ˍˍ9", "ˍ2ˍˍˍ6ˍˍ9", "12ˍˍˍ6ˍˍ9", "ˍˍ3ˍˍ6ˍˍ9", "1ˍ3ˍˍ6ˍˍ9", "ˍ23ˍˍ6ˍˍ9", "123ˍˍ6ˍˍ9",
			"ˍˍˍ4ˍ6ˍˍ9", "1ˍˍ4ˍ6ˍˍ9", "ˍ2ˍ4ˍ6ˍˍ9", "12ˍ4ˍ6ˍˍ9", "ˍˍ34ˍ6ˍˍ9", "1ˍ34ˍ6ˍˍ9", "ˍ234ˍ6ˍˍ9", "1234ˍ6ˍˍ9",
			"ˍˍˍˍ56ˍˍ9", "1ˍˍˍ56ˍˍ9", "ˍ2ˍˍ56ˍˍ9", "12ˍˍ56ˍˍ9", "ˍˍ3ˍ56ˍˍ9", "1ˍ3ˍ56ˍˍ9", "ˍ23ˍ56ˍˍ9", "123ˍ56ˍˍ9",
			"ˍˍˍ456ˍˍ9", "1ˍˍ456ˍˍ9", "ˍ2ˍ456ˍˍ9", "12ˍ456ˍˍ9", "ˍˍ3456ˍˍ9", "1ˍ3456ˍˍ9", "ˍ23456ˍˍ9", "123456ˍˍ9",
			"ˍˍˍˍˍˍ7ˍ9", "1ˍˍˍˍˍ7ˍ9", "ˍ2ˍˍˍˍ7ˍ9", "12ˍˍˍˍ7ˍ9", "ˍˍ3ˍˍˍ7ˍ9", "1ˍ3ˍˍˍ7ˍ9", "ˍ23ˍˍˍ7ˍ9", "123ˍˍˍ7ˍ9",
			"ˍˍˍ4ˍˍ7ˍ9", "1ˍˍ4ˍˍ7ˍ9", "ˍ2ˍ4ˍˍ7ˍ9", "12ˍ4ˍˍ7ˍ9", "ˍˍ34ˍˍ7ˍ9", "1ˍ34ˍˍ7ˍ9", "ˍ234ˍˍ7ˍ9", "1234ˍˍ7ˍ9",
			"ˍˍˍˍ5ˍ7ˍ9", "1ˍˍˍ5ˍ7ˍ9", "ˍ2ˍˍ5ˍ7ˍ9", "12ˍˍ5ˍ7ˍ9", "ˍˍ3ˍ5ˍ7ˍ9", "1ˍ3ˍ5ˍ7ˍ9", "ˍ23ˍ5ˍ7ˍ9", "123ˍ5ˍ7ˍ9",
			"ˍˍˍ45ˍ7ˍ9", "1ˍˍ45ˍ7ˍ9", "ˍ2ˍ45ˍ7ˍ9", "12ˍ45ˍ7ˍ9", "ˍˍ345ˍ7ˍ9", "1ˍ345ˍ7ˍ9", "ˍ2345ˍ7ˍ9", "12345ˍ7ˍ9",
			"ˍˍˍˍˍ67ˍ9", "1ˍˍˍˍ67ˍ9", "ˍ2ˍˍˍ67ˍ9", "12ˍˍˍ67ˍ9", "ˍˍ3ˍˍ67ˍ9", "1ˍ3ˍˍ67ˍ9", "ˍ23ˍˍ67ˍ9", "123ˍˍ67ˍ9",
			"ˍˍˍ4ˍ67ˍ9", "1ˍˍ4ˍ67ˍ9", "ˍ2ˍ4ˍ67ˍ9", "12ˍ4ˍ67ˍ9", "ˍˍ34ˍ67ˍ9", "1ˍ34ˍ67ˍ9", "ˍ234ˍ67ˍ9", "1234ˍ67ˍ9",
			"ˍˍˍˍ567ˍ9", "1ˍˍˍ567ˍ9", "ˍ2ˍˍ567ˍ9", "12ˍˍ567ˍ9", "ˍˍ3ˍ567ˍ9", "1ˍ3ˍ567ˍ9", "ˍ23ˍ567ˍ9", "123ˍ567ˍ9",
			"ˍˍˍ4567ˍ9", "1ˍˍ4567ˍ9", "ˍ2ˍ4567ˍ9", "12ˍ4567ˍ9", "ˍˍ34567ˍ9", "1ˍ34567ˍ9", "ˍ234567ˍ9", "1234567ˍ9",
			"ˍˍˍˍˍˍˍ89", "1ˍˍˍˍˍˍ89", "ˍ2ˍˍˍˍˍ89", "12ˍˍˍˍˍ89", "ˍˍ3ˍˍˍˍ89", "1ˍ3ˍˍˍˍ89", "ˍ23ˍˍˍˍ89", "123ˍˍˍˍ89",
			"ˍˍˍ4ˍˍˍ89", "1ˍˍ4ˍˍˍ89", "ˍ2ˍ4ˍˍˍ89", "12ˍ4ˍˍˍ89", "ˍˍ34ˍˍˍ89", "1ˍ34ˍˍˍ89", "ˍ234ˍˍˍ89", "1234ˍˍˍ89",
			"ˍˍˍˍ5ˍˍ89", "1ˍˍˍ5ˍˍ89", "ˍ2ˍˍ5ˍˍ89", "12ˍˍ5ˍˍ89", "ˍˍ3ˍ5ˍˍ89", "1ˍ3ˍ5ˍˍ89", "ˍ23ˍ5ˍˍ89", "123ˍ5ˍˍ89",
			"ˍˍˍ45ˍˍ89", "1ˍˍ45ˍˍ89", "ˍ2ˍ45ˍˍ89", "12ˍ45ˍˍ89", "ˍˍ345ˍˍ89", "1ˍ345ˍˍ89", "ˍ2345ˍˍ89", "12345ˍˍ89",
			"ˍˍˍˍˍ6ˍ89", "1ˍˍˍˍ6ˍ89", "ˍ2ˍˍˍ6ˍ89", "12ˍˍˍ6ˍ89", "ˍˍ3ˍˍ6ˍ89", "1ˍ3ˍˍ6ˍ89", "ˍ23ˍˍ6ˍ89", "123ˍˍ6ˍ89",
			"ˍˍˍ4ˍ6ˍ89", "1ˍˍ4ˍ6ˍ89", "ˍ2ˍ4ˍ6ˍ89", "12ˍ4ˍ6ˍ89", "ˍˍ34ˍ6ˍ89", "1ˍ34ˍ6ˍ89", "ˍ234ˍ6ˍ89", "1234ˍ6ˍ89",
			"ˍˍˍˍ56ˍ89", "1ˍˍˍ56ˍ89", "ˍ2ˍˍ56ˍ89", "12ˍˍ56ˍ89", "ˍˍ3ˍ56ˍ89", "1ˍ3ˍ56ˍ89", "ˍ23ˍ56ˍ89", "123ˍ56ˍ89",
			"ˍˍˍ456ˍ89", "1ˍˍ456ˍ89", "ˍ2ˍ456ˍ89", "12ˍ456ˍ89", "ˍˍ3456ˍ89", "1ˍ3456ˍ89", "ˍ23456ˍ89", "123456ˍ89",
			"ˍˍˍˍˍˍ789", "1ˍˍˍˍˍ789", "ˍ2ˍˍˍˍ789", "12ˍˍˍˍ789", "ˍˍ3ˍˍˍ789", "1ˍ3ˍˍˍ789", "ˍ23ˍˍˍ789", "123ˍˍˍ789",
			"ˍˍˍ4ˍˍ789", "1ˍˍ4ˍˍ789", "ˍ2ˍ4ˍˍ789", "12ˍ4ˍˍ789", "ˍˍ34ˍˍ789", "1ˍ34ˍˍ789", "ˍ234ˍˍ789", "1234ˍˍ789",
			"ˍˍˍˍ5ˍ789", "1ˍˍˍ5ˍ789", "ˍ2ˍˍ5ˍ789", "12ˍˍ5ˍ789", "ˍˍ3ˍ5ˍ789", "1ˍ3ˍ5ˍ789", "ˍ23ˍ5ˍ789", "123ˍ5ˍ789",
			"ˍˍˍ45ˍ789", "1ˍˍ45ˍ789", "ˍ2ˍ45ˍ789", "12ˍ45ˍ789", "ˍˍ345ˍ789", "1ˍ345ˍ789", "ˍ2345ˍ789", "12345ˍ789",
			"ˍˍˍˍˍ6789", "1ˍˍˍˍ6789", "ˍ2ˍˍˍ6789", "12ˍˍˍ6789", "ˍˍ3ˍˍ6789", "1ˍ3ˍˍ6789", "ˍ23ˍˍ6789", "123ˍˍ6789",
			"ˍˍˍ4ˍ6789", "1ˍˍ4ˍ6789", "ˍ2ˍ4ˍ6789", "12ˍ4ˍ6789", "ˍˍ34ˍ6789", "1ˍ34ˍ6789", "ˍ234ˍ6789", "1234ˍ6789",
			"ˍˍˍˍ56789", "1ˍˍˍ56789", "ˍ2ˍˍ56789", "12ˍˍ56789", "ˍˍ3ˍ56789", "1ˍ3ˍ56789", "ˍ23ˍ56789", "123ˍ56789",
			"ˍˍˍ456789", "1ˍˍ456789", "ˍ2ˍ456789", "12ˍ456789", "ˍˍ3456789", "1ˍ3456789", "ˍ23456789", "123456789",
		]
		codes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

		data = np.zeros((9, 9), dtype = np.uint16)
		for head_id in self.scanners[0]:
			for item_id in self.heads[head_id]:
				x, y, d = self.nodes_info[item_id]
				data[x, y] += codes[d]

		tmp = "╔═══════╤═══════╤═══════╤═══════╤═══════╤═══════╤═══════╤═══════╤═══════╗\n"
		for i in range(9):
			if i > 0:
				tmp += "║"
				for j in range(9):
					if j > 0:
						code = 0
						if self.xy2uv[i - 1][j - 1][0] != self.xy2uv[i - 1][j][0]:
							code += 1
						if self.xy2uv[i - 1][j - 1][0] != self.xy2uv[i][j - 1][0]:
							code += 2
						if self.xy2uv[i - 1][j][0] != self.xy2uv[i][j][0]:
							code += 4
						if self.xy2uv[i][j - 1][0] != self.xy2uv[i][j][0]:
							code += 8
						tmp += "+++╝+╚═╩+║╗╣╔╠╦╬"[code]
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i - 1][j][0]
					if m1 == m2:
						tmp += "⋅⋅⋅⋅⋅⋅⋅"
					else:
						tmp += "═══════"
				tmp += "║\n"

			tmp += "║  "
			for j in range(9):
				if j > 0:
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i][j - 1][0]
					if m1 == m2:
						tmp += "  :  "
					else:
						tmp += "  ║  "
				if self.result[i][j] > 0:
					tmp += "   "
				else:
					s = code2str[data[i, j]]
					tmp += s[0:3]
				pass
			tmp += "  ║\n║  "
			for j in range(9):
				if j > 0:
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i][j - 1][0]
					if m1 == m2:
						tmp += "  :  "
					else:
						tmp += "  ║  "
				if self.result[i][j] > 0:
					tmp += f" {self.result[i][j]} "
				else:
					s = code2str[data[i, j]]
					tmp += s[3:6]
				pass
			tmp += "  ║\n║  "
			for j in range(9):
				if j > 0:
					m1 = self.xy2uv[i][j][0]
					m2 = self.xy2uv[i][j - 1][0]
					if m1 == m2:
						tmp += "  :  "
					else:
						tmp += "  ║  "
				if self.result[i][j] > 0:
					tmp += "   "
				else:
					s = code2str[data[i, j]]
					tmp += s[6:9]
				pass
			tmp += "  ║\n"
		tmp += "╚═══════╧═══════╧═══════╧═══════╧═══════╧═══════╧═══════╧═══════╧═══════╝"
		return tmp

	def str_result(self):
		tmp = "╔═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╗\n"
		for i in range(9):
			if i > 0:
				tmp += "║"
				for j in range(9):
					if j > 0:
						code = 0
						if self.xy2uv[i - 1][j - 1][0] != self.xy2uv[i - 1][j][0]:
							code += 1
						if self.xy2uv[i - 1][j - 1][0] != self.xy2uv[i][j - 1][0]:
							code += 2
						if self.xy2uv[i - 1][j][0] != self.xy2uv[i][j][0]:
							code += 4
						if self.xy2uv[i][j - 1][0] != self.xy2uv[i][j][0]:
							code += 8
						tmp += "+++╝+╚═╩+║╗╣╔╠╦╬"[code]
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

	@staticmethod
	def str_data(data):
		tmp_str = ""
		for i in range(9):
			if i > 0 and i % 3 == 0:
				tmp_str += " ------+-------+------\n"
			for j in range(9):
				if j > 0 and j % 3 == 0: tmp_str += " |"
				if data[i][j] == 0:
					tmp_str += " ."
				else:
					tmp_str += f" {data[i][j]}"
			tmp_str += "\n"
		return tmp_str


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


	def read(path):
		z = 0
		data = [0] * 162
		block = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(9)]
		board = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(9)]
		with open(path, "r") as fr:
			for line in fr:
				for c in line:
					if z >= 162:
						break
					if c == ".":
						data[z] = 0
						z += 1
					elif "0" <= c <= "9":
						data[z] = int(c) - int("0")
						z += 1
		for z, d in enumerate(data[:81]):
			i = z // 9
			j = z % 9
			block[i][j] = d
		for z, d in enumerate(data[81:]):
			i = z // 9
			j = z % 9
			board[i][j] = d
		return block, board


	def main():
		if len(sys.argv) == 1: max_result = 16
		else: max_result = int(sys.argv[1])

		folder = os.path.dirname(os.path.abspath(__file__))
		file = folder + "/sudoku.txt"
		# os.system("touch \"" + file + "\"")
		# os.system("open \"" + file + "\"")
		# print(">> press any key to continue ..")
		# input()
		block, board = read(file)

		su = Sudoku()
		su.setup(board, block)
		su.solve(max_result)

		with open(file, "w") as fw:
			for tmp_str in su.logs:
				fw.write(tmp_str)
				fw.write("\n")


	main()
	pass
