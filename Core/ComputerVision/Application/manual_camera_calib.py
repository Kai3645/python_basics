import math
import os
import re
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

if __name__ == "__main__":
	FOLDER = "/home/lab/Desktop/python_resource/M24_12/D2412_20/res/"
	UPDATE_share = False
	
	
	def circle_mask(W, H, K = 1.0):
		a_w = np.arange(W)
		a_h = np.arange(H)
		a_ww, a_hh = np.meshgrid(a_w, a_h)
		pts = np.vstack([a_hh.ravel(), a_ww.ravel()]).T
		mask = np.zeros((H, W), dtype = np.uint8)
		c = (H / 2, W / 2)
		r = min(W / 2, H / 2) * K
		diff = pts - c
		dd = np.sum(diff * diff, axis = 1)
		mask[pts[dd < r * r]] = 255
		return mask
	
	
	def hist(X, th, N = 20):
		center = np.average(X)
		valid = None
		for i in range(N):
			valid = np.abs(X - center) < th
			c_tmp = np.average(X[valid])
			if abs(c_tmp - center) < 0.1: break
			center = c_tmp
		return center, valid
	
	
	def detect_center_cross(image, mask, th = 10):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (3, 3), math.sqrt(2))
		_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		kernel = np.ones((3, 3), np.uint8)
		img = cv2.dilate(img, kernel, iterations = 1)
		img = cv2.erode(img, kernel, iterations = 2)
		
		img[mask == 0] = 0
		
		pts = np.argwhere(img > 200)
		
		cx, v1 = hist(pts[:, 0], th)
		cy, v2 = hist(pts[:, 1], th)
		
		def fitting(X):
			_ave = np.mean(X, axis = 0)
			_X = X - _ave
			_V = np.dot(_X.T, _X) / (len(X) - 1)
			_u, _, _ = np.linalg.svd(_V)
			
			_nv = _u[:, -1]
			_nv /= np.linalg.norm(_nv)
			_d = np.dot(_nv, _ave)
			if _d < 0: return -_nv, -_d
			return _nv, _d
		
		nv1, d1 = fitting(pts[v1])
		nv2, d2 = fitting(pts[v2])
		
		cx = (nv2[1] * d1 - nv1[1] * d2) / (nv1[0] * nv2[1] - nv1[1] * nv2[0])
		cy = (nv1[0] * d2 - nv2[0] * d1) / (nv1[0] * nv2[1] - nv1[1] * nv2[0])
		return cx, cy
	
	
	class CNode:
		def __init__(self, pos):
			self.is_edge = False
			self.pos = pos
			self.N0 = None
			self.N1 = None
			self.N2 = None
			self.N3 = None
			pass
		
		def set_sub(self, N0, N1, N2, N3):
			self.N0 = N0
			self.N1 = N1
			self.N2 = N2
			self.N3 = N3
			pass
		
		def get_pos(self):
			return float(self.pos[0]), float(self.pos[1])
		
		def get_corners(self):
			if not self.is_edge: return None
			P0 = self.N0.get_pos()
			P1 = self.N1.get_pos()
			P2 = self.N2.get_pos()
			P3 = self.N3.get_pos()
			return np.asarray([P0, P1, P2, P3], np.float32)
		
		def get_roi(self):
			pts = self.get_corners()
			h0 = int(np.min(pts[:, 0]))
			w0 = int(np.min(pts[:, 1]))
			h1 = math.ceil(np.max(pts[:, 0]))
			w1 = math.ceil(np.max(pts[:, 1]))
			return h0, w0, h1, w1
		
		def detect(self, W, H, image):
			pts = self.get_corners()
			x0 = int(np.min(pts[:, 0]))
			y0 = int(np.min(pts[:, 1]))
			x1 = math.ceil(np.max(pts[:, 0]))
			y1 = math.ceil(np.max(pts[:, 1]))
			
			P_src = pts - (x0, y0)
			P_dst = np.asarray([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
			mat = cv2.getPerspectiveTransform(P_src, P_dst)
			dst = cv2.warpPerspective(image[x0:x1, y0:y1], mat, W, H, flags = cv2.INTER_CUBIC,
			                          borderMode = cv2.BORDER_REFLECT)
			mask = circle_mask(W, H, 0.9)
			C = detect_center_cross(dst, mask, th = 10)
			
			inv_mat = np.linalg.inv(mat)
			den = float(inv_mat[2, 0] * C[0] + inv_mat[2, 1] * C[1] + inv_mat[2, 2])
			cx = float((inv_mat[0, 0] * C[0] + inv_mat[0, 1] * C[1] + inv_mat[0, 2]) / den)
			cy = float((inv_mat[1, 0] * C[0] + inv_mat[1, 1] * C[1] + inv_mat[1, 2]) / den)
			return cx, cy
	
	
	class DisplayCanvas(tk.Canvas):
		MAGNIFICATION = math.sqrt(2)
		MIN_PIXEL = 200
		MAX_PIXEL = 20000
		PPM = 5000
		
		def __init__(self, win, width, height):
			self.is_init = False
			self.can_width = width
			self.can_height = height
			
			self.left_pos = [0, 0]
			self.right_pos = [0, 0]
			self.middle_pos = [0, 0]
			self.left_press = False
			self.right_press = False
			self.middle_press = False
			
			self.level = 0
			self.min_level = 0
			self.max_level = 0
			self.img_offset = [0, 0]
			self.img_size = [0, 0]
			self.loc_img = None
			self.tk_img = None
			self.data_dict = None
			
			self.tk_rect = None
			self.tk_grid = None
			
			self.corners = np.zeros((4, 2), float)
			
			super().__init__(win, relief = tk.FLAT, width = width, height = height)
			
			# 鼠标事件报告
			self.label_e = tk.Label(self, text = "Helo event .. ")
			self.label_e.place(x = 8, y = 8)
		
		def init_image(self, image):
			H, W = image.shape[:2]
			
			# self.img_offset[:] = [0, 0]
			# self.img_size[:] = [W, H]
			
			if self.is_init is not True:
				self.is_init = True
			
			self.corners[:, :] = [[0, 0], [W, 0], [W, H], [0, H]]
			
			self.data_dict = dict()
			self.data_dict = {0: (W, H, image)}
			
			tmp_w = W
			self.min_level = 0
			while tmp_w / self.MAGNIFICATION > self.MIN_PIXEL:
				new_level = self.min_level - 1
				ratio = math.pow(self.MAGNIFICATION, new_level)
				tmp_w = round(W * ratio)
				tmp_h = round(H * ratio)
				img = self.data_dict[self.min_level][2]
				img = cv2.resize(img, (tmp_w, tmp_h), interpolation = cv2.INTER_LINEAR)
				self.data_dict[new_level] = (tmp_w, tmp_h, img)
				self.min_level = new_level
			
			tmp_w = W
			self.max_level = 0
			while tmp_w * self.MAGNIFICATION < self.MAX_PIXEL:
				new_level = self.max_level + 1
				ratio = math.pow(self.MAGNIFICATION, new_level)
				tmp_w = round(W * ratio)
				tmp_h = round(H * ratio)
				img = self.data_dict[self.max_level][2]
				img = cv2.resize(img, (tmp_w, tmp_h), interpolation = cv2.INTER_LINEAR)
				self.data_dict[new_level] = (tmp_w, tmp_h, img)
				self.max_level = new_level
			
			w, h, img = self.data_dict[self.level]
			self.img_size[:] = w, h
			self.loc_img = img
			
			self.bind("<Enter>", self.mouse_enter)
			self.bind("<Leave>", self.mouse_leave)
			self.bind("<Motion>", self.mouse_move)
			
			self.bind("<Button-4>", self.mouse_wheel_up)
			self.bind("<Button-5>", self.mouse_wheel_down)
			
			self.bind("<ButtonPress-1>", self.mouse_press_left)
			self.bind("<ButtonPress-2>", self.mouse_press_middle)
			self.bind("<ButtonPress-3>", self.mouse_press_right)
			
			self.bind("<ButtonRelease-1>", self.mouse_release_left)
			self.bind("<ButtonRelease-2>", self.mouse_release_middle)
			self.bind("<ButtonRelease-3>", self.mouse_release_right)
			
			self.bind("<Double-Button-1>", self.mouse_double_left)
			self.bind("<Double-Button-2>", self.mouse_double_middle)
			self.bind("<Double-Button-3>", self.mouse_double_right)
			
			self.bind("<B1-Motion>", self.mouse_drag_left)
			self.bind("<B2-Motion>", self.mouse_drag_middle)
			self.bind("<B3-Motion>", self.mouse_drag_right)
			
			self.update()
		
		# 鼠标进入
		def mouse_enter(self, event):
			self.label_e["text"] = str(event)
		
		# 鼠标离开
		def mouse_leave(self, event):
			self.label_e["text"] = str(event)
		
		# 鼠标移动
		def mouse_move(self, event):
			self.label_e["text"] = str(event)
		
		@staticmethod
		def pos_calc(x, cx, k):
			x = k * (x - cx) + cx
			return x
		
		# 鼠标滚轮
		def mouse_wheel_up(self, event):
			self.label_e["text"] = str(event)
			if self.level < self.max_level:
				self.level += 1
				w, h, self.loc_img = self.data_dict[self.level]
				cx = self.can_width / 2
				cy = self.can_height / 2
				k = self.MAGNIFICATION
				self.img_offset[0] = round(self.pos_calc(self.img_offset[0], cx, k))
				self.img_offset[1] = round(self.pos_calc(self.img_offset[1], cy, k))
				self.img_size[:] = w, h
				
				self.update()
		
		def mouse_wheel_down(self, event):
			self.label_e["text"] = str(event)
			if self.level > self.min_level:
				self.level -= 1
				w, h, self.loc_img = self.data_dict[self.level]
				cx = self.can_width / 2
				cy = self.can_height / 2
				k = 1 / self.MAGNIFICATION
				self.img_offset[0] = round(self.pos_calc(self.img_offset[0], cx, k))
				self.img_offset[1] = round(self.pos_calc(self.img_offset[1], cy, k))
				self.img_size[:] = w, h
				
				self.update()
		
		# 鼠标按下
		def mouse_press_left(self, event):
			self.label_e["text"] = str(event)
			self.left_press = True
			self.left_pos[:] = event.x, event.y
			self.focus_set()
		
		def mouse_press_middle(self, event):
			self.label_e["text"] = str(event)
			self.middle_press = True
			self.middle_pos[:] = event.x, event.y
			self.focus_set()
		
		def mouse_press_right(self, event):
			self.label_e["text"] = str(event)
			self.right_press = True
			self.right_pos[:] = event.x, event.y
			self.focus_set()
		
		# 鼠标释放
		def mouse_release_left(self, event):
			self.label_e["text"] = str(event)
			self.left_press = False
		
		def mouse_release_middle(self, event):
			self.label_e["text"] = str(event)
			self.middle_press = False
		
		def mouse_release_right(self, event):
			self.label_e["text"] = str(event)
			self.right_press = False
		
		# 鼠标双击
		def mouse_double_left(self, event):
			self.focus_set()
			self.label_e["text"] = str(event)
		
		def mouse_double_middle(self, event):
			self.focus_set()
			self.label_e["text"] = str(event)
		
		def mouse_double_right(self, event):
			self.focus_set()
			self.label_e["text"] = str(event)
		
		# 鼠标拖拽
		def mouse_drag_left(self, event):
			self.label_e["text"] = str(event)
		
		def mouse_drag_middle(self, event):
			self.label_e["text"] = str(event)
			self.img_offset[0] += event.x - self.middle_pos[0]
			self.img_offset[1] += event.y - self.middle_pos[1]
			self.middle_pos[:] = event.x, event.y
			
			self.update()
		
		def mouse_drag_right(self, event):
			self.label_e["text"] = str(event)
		
		def tk_delete(self):
			self.delete_grid()
			self.delete(self.tk_rect)
		
		def get_loc_corners(self):
			ratio = math.pow(self.MAGNIFICATION, self.level)
			return self.corners * ratio + self.img_offset
		
		def create_grid(self, a: int, b: int):
			loc_corners = self.get_loc_corners()
			delta_up = (loc_corners[1] - loc_corners[0]) / a
			delta_down = (loc_corners[2] - loc_corners[3]) / a
			delta_left = (loc_corners[3] - loc_corners[0]) / b
			delta_right = (loc_corners[2] - loc_corners[1]) / b
			
			self.tk_grid = []
			for i in range(1, a):
				x0, y0 = loc_corners[0] + delta_up * i
				x1, y1 = loc_corners[3] + delta_down * i
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "white"))
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "black", dash = [5, 5]))
			for i in range(1, b):
				x0, y0 = loc_corners[0] + delta_left * i
				x1, y1 = loc_corners[1] + delta_right * i
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "white"))
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "black", dash = [5, 5]))
			points = [
				(float(loc_corners[0, 0]), float(loc_corners[0, 1])),
				(float(loc_corners[1, 0]), float(loc_corners[1, 1])),
				(float(loc_corners[2, 0]), float(loc_corners[2, 1])),
				(float(loc_corners[3, 0]), float(loc_corners[3, 1])),
				(float(loc_corners[0, 0]), float(loc_corners[0, 1])),
			]
			self.tk_rect = self.create_line(points, width = 1, fill = "orange red")
		
		def delete_grid(self):
			if self.tk_grid is not None:
				for tl in self.tk_grid:
					self.delete(tl)
			self.delete(self.tk_rect)
		
		def update_image(self):
			x0, y0 = self.img_offset
			x1, y1 = x0 + self.img_size[0], y0 + self.img_size[1]
			
			if x0 >= self.can_width: return
			if y0 >= self.can_height: return
			if x1 <= 0: return
			if y1 <= 0: return
			
			if x0 < 0: w0, a = -x0, 0
			else: w0, a = 0, x0
			if y0 < 0: h0, b = -y0, 0
			else: h0, b = 0, y0
			
			if x1 < self.can_width: w1 = self.img_size[0]
			else: w1 = self.can_width - x0
			if y1 < self.can_height: h1 = self.img_size[1]
			else: h1 = self.can_height - y0
			
			img = self.loc_img[h0:h1, w0:w1, :]
			self.tk_img = ImageTk.PhotoImage(Image.fromarray(img))
			self.create_image(a, b, image = self.tk_img, anchor = "nw")
		
		def update(self):
			self.update_image()
			
			self.tk_delete()
			self.create_grid(10, 10)
			
			pass
	
	
	class MainCanvas(DisplayCanvas):
		def __init__(self, win, w, h):
			super().__init__(win, w, h)
			
			self.tk_ellipse = None
			self.tk_line = None
			self.corner_id = 0
			
			self.grid_wn = 10
			self.grid_hn = 10
			self.grid_wa = 10
			self.grid_ha = 10
			
			self.is_edge_locked = False
			self.node_map = []
		
		def set_nodes(self):
			node_map = []
			loc_corners = self.get_loc_corners()
			
			corner_src = loc_corners[:4].astype(np.float32)
			corner_dst = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
			mat = cv2.getPerspectiveTransform(corner_dst, corner_src)
			
			for i in range(self.grid_wn):
				row_map = []
				for j in range(self.grid_hn):
					xi = i / self.grid_wn
					yi = j / self.grid_hn
					
					den = float(mat[2, 0] * xi + mat[2, 1] * yi + mat[2, 2])
					xi = float((mat[0, 0] * xi + mat[0, 1] * yi + mat[0, 2]) / den)
					yi = float((mat[1, 0] * xi + mat[1, 1] * yi + mat[1, 2]) / den)
					
					tmp_node = CNode((xi, yi))
					if i == 0 or i == (self.grid_wn - 1):
						tmp_node.is_edge = True
					if j == 0 or j == (self.grid_hn - 1):
						tmp_node.is_edge = True
					row_map.append(tmp_node)
				node_map.append(row_map)
			
			for i in range(1, self.grid_wn - 1):
				for j in range(1, self.grid_hn - 1):
					node_map[i][j].set_sub(
						node_map[i - 1][j - 1],
						node_map[i - 1][j + 1],
						node_map[i + 1][j + 1],
						node_map[i + 1][j - 1],
					)
			
			return node_map
		
		def mouse_press_left(self, event):
			super().mouse_press_left(event)
			
			loc_corners = self.get_loc_corners()
			diff = loc_corners - (event.x, event.y)
			d = np.sum(diff * diff, axis = 1)
			d[4:] *= 5
			self.corner_id = np.argmin(d)
			
			self.update()
		
		def mouse_release_left(self, event):
			super().mouse_release_left(event)
			
			self.delete(self.tk_line)
		
		def mouse_drag_left(self, event):
			self.label_e["text"] = str(event)
			ratio = math.pow(self.MAGNIFICATION, self.level)
			diff_x = (event.x - self.left_pos[0]) / ratio
			diff_y = (event.y - self.left_pos[1]) / ratio
			self.left_pos[:] = event.x, event.y
			if self.corner_id < 4:
				self.corners[self.corner_id] += diff_x, diff_y
			elif self.corner_id == 4:
				self.corners[[0, 1]] += diff_x, diff_y
			elif self.corner_id == 5:
				self.corners[[1, 2]] += diff_x, diff_y
			elif self.corner_id == 6:
				self.corners[[2, 3]] += diff_x, diff_y
			elif self.corner_id == 7:
				self.corners[[3, 0]] += diff_x, diff_y
			
			self.update()
		
		def tk_delete(self):
			super().tk_delete()
			
			self.delete(self.tk_line)
			self.delete(self.tk_ellipse)
		
		def update(self):
			self.update_image()
			
			self.tk_delete()
			
			self.create_grid(self.grid_wn, self.grid_hn)
			
			loc_corners = self.get_loc_corners()
			if self.left_press:
				x0, y0 = loc_corners[self.corner_id]
				x1, y1 = self.left_pos
				x0 = self.pos_calc(x0, x1, 0.95)
				y0 = self.pos_calc(y0, y1, 0.95)
				self.tk_line = self.create_line(x0, y0, x1, y1, width = 1, fill = "plum1", dash = [12, 4, 4, 4])
			points = [
				(float(loc_corners[0, 0]), float(loc_corners[0, 1])),
				(float(loc_corners[1, 0]), float(loc_corners[1, 1])),
				(float(loc_corners[2, 0]), float(loc_corners[2, 1])),
				(float(loc_corners[3, 0]), float(loc_corners[3, 1])),
				(float(loc_corners[0, 0]), float(loc_corners[0, 1])),
			]
			self.tk_rect = self.create_line(points, width = 1, fill = "orange red")
			self.tk_ellipse = self.create_oval(
				float(loc_corners[0, 0]) - 20, float(loc_corners[0, 1]) - 20,
				float(loc_corners[0, 0]) + 20, float(loc_corners[0, 1]) + 20,
				width = 1, outline = "tomato",
			)
		
		def get_crop(self, width_real, height_real):
			w = round(width_real * self.PPM / 1000)
			h = round(height_real * self.PPM / 1000)
			
			corner_src = np.copy(self.corners).astype(np.float32)
			corner_dst = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
			
			mat = cv2.getPerspectiveTransform(corner_src, corner_dst)
			
			src = self.data_dict[0][2]
			dst = cv2.warpPerspective(src, mat, (w, h), flags = cv2.INTER_CUBIC,
			                          borderMode = cv2.BORDER_REFLECT)
			return dst
			
			pass
		
		def get_loc_corners(self):
			ratio = math.pow(self.MAGNIFICATION, self.level)
			P0, P1, P2, P3 = self.corners * ratio + self.img_offset
			return np.asarray([
				P0, P1, P2, P3,
				(P0 + P1) / 2,
				(P1 + P2) / 2,
				(P2 + P3) / 2,
				(P3 + P0) / 2,
			], dtype = float)
		
		def create_grid(self, a: int, b: int):
			self.tk_grid = []
			for i in range(1, self.grid_wn):
				for j in range(1, self.grid_hn):
					x0, y0 = self.node_map[i][j].get_pos()
					print(x0, y0)
					if i < self.grid_wn - 1:
						x1, y1 = self.node_map[i][j - 1].get_pos()
						self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "white"))
						self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "black", dash = [5, 5]))
					if j < self.grid_hn - 1:
						x1, y1 = self.node_map[i - 1][j].get_pos()
						self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "white"))
						self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "black", dash = [5, 5]))
			pass
		
		def set_grid_info(self, wn, hn, wa, ha):
			self.grid_wn = wn
			self.grid_hn = hn
			self.grid_wa = wa
			self.grid_ha = ha
			if self.is_init:
				self.update()
	
	
	class Application:
		FRAME1_WIDTH = 1200
		FRAME1_HEIGHT = 1000
		FRAME2_WIDTH = 420
		
		def __init__(self):
			self.win = tk.Tk()
			self.win.title("Manual Calibration")
			self.win.geometry(f"{self.FRAME1_WIDTH + self.FRAME2_WIDTH}x{self.FRAME1_HEIGHT}+8+8")
			self.win.resizable(False, False)
			
			self.frame_1 = tk.Frame(self.win, relief = tk.FLAT,
			                        width = self.FRAME1_WIDTH,
			                        height = self.FRAME1_HEIGHT)
			self.frame_1.grid(row = 0, column = 0, sticky = tk.NW, )
			self.frame_2 = tk.Frame(self.win, relief = tk.FLAT,
			                        width = self.FRAME2_WIDTH,
			                        height = self.FRAME1_HEIGHT)
			self.frame_2.grid(row = 0, column = 1, sticky = tk.NW)
			
			self.can = MainCanvas(self.frame_1, self.FRAME1_WIDTH, self.FRAME1_HEIGHT)
			self.can.place(x = 0, y = 0)
			
			def is_positive_integer(s):
				if re.match(re.compile(r"\d"), s):
					return True
				else:
					return False
			
			def is_positive_float(s):
				if re.match(re.compile(r"[\d.]"), s):
					return True
				else:
					return False
			
			tk.Label(self.frame_2, text = "水平 /格", width = 8).grid(row = 0, column = 0)
			self.E_w_n = tk.Entry(self.frame_2, width = 4, justify = "center", validate = 'key',
			                      validatecommand = (self.frame_2.register(is_positive_integer), "%S"))
			self.E_w_n.grid(row = 0, column = 1)
			self.E_w_n.insert(tk.END, "10")
			tk.Label(self.frame_2, text = "单位 mm/格", width = 12).grid(row = 0, column = 2)
			self.E_w_a = tk.Entry(self.frame_2, width = 12, justify = "left", validate = 'key',
			                      validatecommand = (self.frame_2.register(is_positive_float), "%S"))
			self.E_w_a.grid(row = 0, column = 3)
			self.E_w_a.insert(tk.END, "10")
			
			tk.Label(self.frame_2, text = "垂直 /格", width = 8).grid(row = 1, column = 0)
			self.E_h_n = tk.Entry(self.frame_2, width = 4, justify = "center", validate = 'key',
			                      validatecommand = (self.frame_2.register(is_positive_integer), "%S"))
			self.E_h_n.grid(row = 1, column = 1)
			self.E_h_n.insert(tk.END, "10")
			tk.Label(self.frame_2, text = "单位 mm/格", width = 12).grid(row = 1, column = 2)
			self.E_h_a = tk.Entry(self.frame_2, width = 12, justify = "left", validate = 'key',
			                      validatecommand = (self.frame_2.register(is_positive_float), "%S"))
			self.E_h_a.grid(row = 1, column = 3)
			self.E_h_a.insert(tk.END, "10")
			# self.frame_2.grid_rowconfigure(0, minsize = 40)
			# self.frame_2.grid_rowconfigure(1, minsize = 40)
			self.frame_2.grid_columnconfigure(3, minsize = 130)
			
			def copy_upper():
				try:
					wa = float(self.E_w_a.get())
					self.E_h_a.delete(0, tk.END)
					self.E_h_a.insert(tk.END, str(wa))
				except ValueError:
					pass
			
			B_tmp = tk.Button(self.frame_2, text = "等边", width = 2, command = copy_upper)
			B_tmp.grid(row = 0, column = 4, sticky = tk.S)
			
			def get_entry_data():
				try:
					wn = int(self.E_w_n.get())
					hn = int(self.E_h_n.get())
					wa = float(self.E_w_a.get())
					ha = float(self.E_h_a.get())
					return True, wn, hn, wa, ha
				except ValueError:
					return False, 0, 0, 0, 0
			
			self.is_lock_entry = False
			
			def Lock():
				if self.is_lock_entry:
					self.is_lock_entry = False
					self.E_w_n.config(state = "normal")
					self.E_w_a.config(state = "normal")
					self.E_h_n.config(state = "normal")
					self.E_h_a.config(state = "normal")
				else:
					flag, wn, hn, wa, ha = get_entry_data()
					if flag is False: return
					self.is_lock_entry = True
					self.E_w_n.config(state = "readonly")
					self.E_w_a.config(state = "readonly")
					self.E_h_n.config(state = "readonly")
					self.E_h_a.config(state = "readonly")
					self.can.set_grid_info(wn, hn, wa, ha)
			
			B_tmp = tk.Button(self.frame_2, text = "锁定", width = 2, command = Lock)
			B_tmp.grid(row = 1, column = 4, sticky = tk.S)
			
			loc_frame = tk.Frame(self.frame_2, relief = tk.FLAT)
			loc_frame.grid(row = 2, column = 0, columnspan = 4, sticky = tk.W)
			loc_frame.grid_rowconfigure(0, minsize = 50)
			
			B_import = tk.Button(loc_frame, text = "添加", command = self.get_image_path, justify = tk.CENTER)
			B_import.grid(row = 0, column = 0, sticky = tk.SE)
			loc_frame.grid_columnconfigure(0, minsize = 64)
			B_output = tk.Button(loc_frame, text = "锁边", justify = tk.CENTER)
			B_output.grid(row = 0, column = 1, sticky = tk.SE)
			loc_frame.grid_columnconfigure(1, minsize = 64)
			B_update = tk.Button(loc_frame, text = "更新", justify = tk.CENTER)
			B_update.grid(row = 0, column = 2, sticky = tk.SE)
			loc_frame.grid_columnconfigure(2, minsize = 64)
			
			T_info = tk.Text(self.frame_2, width = 50, height = 48)
			T_info.config(state = "disabled")
			T_info.grid(row = 3, column = 0, columnspan = 8, sticky = tk.N)
			
			self.working_folder = ""
			self.working_name = ""
			
			pass
		
		def get_image_path(self):
			img_exts = r"*.jpeg  *.jpg *.png"
			
			path = filedialog.askopenfilename(
				filetypes = [("image", img_exts)],
				initialdir = FOLDER,
			)
			if type(path) is not str: return
			if path == "": return
			try:
				img = cv2.imread(path, cv2.IMREAD_COLOR)
				if img is not None:
					self.working_folder = os.path.dirname(path) + os.path.sep
					self.working_name = os.path.basename(path)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					self.can.init_image(img)
			except FileNotFoundError:
				return
		
		# def save_image(self):
		# 	w_real, h_real = self.get_image_real_size()
		# 	img = self.can_1.get_crop(w_real, h_real)
		#
		# 	path = self.working_folder + "warped_" + self.working_name
		# 	cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		
		def run(self):
			self.win.mainloop()
		
		# def update(self):
		# 	self.W_entry.config(state = "readonly")
		# 	self.H_entry.config(state = "readonly")
		# 	if self.can_1.is_init is True:
		# 		w, h = self.get_image_real_size()
		# 		img = self.can_1.get_crop(w, h)
		# 		self.can_2.init_image(img)
		# 		self.can_2.update()
		
		pass
	
	
	def main():
		app = Application()
		app.run()
	
	
	main()
	
	pass
