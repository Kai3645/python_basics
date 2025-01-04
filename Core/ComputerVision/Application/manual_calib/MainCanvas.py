import math
import os.path
import sys

import cv2
import numpy as np

from DisplayCanvas import DisplayCanvas
from Node import CNode, circle_mask


class MainCanvas(DisplayCanvas):
	PPMM = 10  # pixel/mm
	DETECT_THRESHOLD = round(PPMM * 0.35)
	
	def __init__(self, win, width, height):
		super().__init__(win, width, height)
		
		self.tk_line = None
		self.tk_rect = None
		self.tk_grid = None
		self.tk_ellipse = None
		
		self.grid_wn = 3
		self.grid_hn = 3
		self.grid_wa = 10
		self.grid_ha = 10
		
		self.node_map = None
		self.corners = None
		self.targ_id = 0
		
		self.is_edge_locked = False
		
		pass
	
	def init_image(self, image):
		h, w = image.shape[:2]
		self.corners = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
		
		self.init_nodes()
		
		super().init_image(image)
		pass
	
	def init_grid_info(self, wn, hn, wa, ha):
		self.grid_wn = wn
		self.grid_hn = hn
		self.grid_wa = wa
		self.grid_ha = ha
		
		self.init_nodes()
		self.update()
	
	def init_nodes(self):
		corners_dst = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
		mat = cv2.getPerspectiveTransform(corners_dst, self.corners)
		
		node_map = []
		for i in range(self.grid_wn + 1):
			row_map = []
			for j in range(self.grid_hn + 1):
				xi = i / self.grid_wn
				yi = j / self.grid_hn
				den = mat[2, 0] * xi + mat[2, 1] * yi + mat[2, 2]
				xj = float((mat[0, 0] * xi + mat[0, 1] * yi + mat[0, 2]) / den)
				yj = float((mat[1, 0] * xi + mat[1, 1] * yi + mat[1, 2]) / den)
				tmp_node = CNode((xj, yj))
				row_map.append(tmp_node)
			node_map.append(row_map)
		for i in range(1, self.grid_wn):
			for j in range(1, self.grid_hn):
				node_map[i][j].set_sub(
					node_map[i - 1][j - 1],
					node_map[i + 1][j - 1],
					node_map[i + 1][j + 1],
					node_map[i - 1][j + 1],
				)
				pass
			pass
		self.node_map = node_map
		
		pass
	
	def update_nodes(self, new_corners):
		mat = cv2.getPerspectiveTransform(self.corners, new_corners)
		for i in range(1, self.grid_wn):
			for j in range(1, self.grid_hn):
				xi, yi = self.node_map[i][j].get_pos()
				den = mat[2, 0] * xi + mat[2, 1] * yi + mat[2, 2]
				xj = float((mat[0, 0] * xi + mat[0, 1] * yi + mat[0, 2]) / den)
				yj = float((mat[1, 0] * xi + mat[1, 1] * yi + mat[1, 2]) / den)
				self.node_map[i][j].set_pos(xj, yj)
				pass
			pass
		corners_dst = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
		mat = cv2.getPerspectiveTransform(corners_dst, new_corners)
		for i in range(self.grid_wn + 1):
			for j in range(self.grid_hn + 1):
				if 0 < i < self.grid_wn and 0 < j < self.grid_hn: continue
				xi = i / self.grid_wn
				yi = j / self.grid_hn
				den = mat[2, 0] * xi + mat[2, 1] * yi + mat[2, 2]
				xj = float((mat[0, 0] * xi + mat[0, 1] * yi + mat[0, 2]) / den)
				yj = float((mat[1, 0] * xi + mat[1, 1] * yi + mat[1, 2]) / den)
				self.node_map[i][j].set_pos(xj, yj)
				pass
			pass
		self.corners[:, :] = new_corners
		pass
	
	def calc_closest_point_id(self, x, y):
		ratio = math.pow(self.MAGNIFICATION, self.level)
		if not self.is_edge_locked:
			pts = np.asarray(self.corners)
		else:
			pts = np.asarray([[n.get_pos() for n in row] for row in self.node_map]).reshape((-1, 2))
		delta = pts * ratio + self.loc_offset - (x, y)
		d = np.sum(delta * delta, axis = 1)
		targ_id = np.argmin(d)
		self.targ_id = targ_id
		pass
	
	def mouse_press_left(self, event):
		super().mouse_press_left(event)
		self.calc_closest_point_id(event.x, event.y)
		
		self.update()
		pass
	
	def mouse_release_left(self, event):
		super().mouse_release_left(event)
		
		self.delete(self.tk_line)
		pass
	
	def corner_drag_check(self, dx, dy):
		i = (self.targ_id + 1) % 4
		j = (self.targ_id + 2) % 4
		k = (self.targ_id + 3) % 4
		p0 = self.corners[self.targ_id] + (dx, dy)
		vx1, vy1 = self.corners[j] - p0
		vx2, vy2 = self.corners[k] - self.corners[i]
		w0j = (vy1, -vx1)
		wik = (vy2, -vx2)
		v0i = self.corners[i] - p0
		v0k = self.corners[k] - p0
		vji = self.corners[i] - self.corners[j]
		flag1 = np.dot(w0j, v0i)
		flag2 = np.dot(w0j, v0k)
		flag3 = np.dot(wik, v0i)
		flag4 = np.dot(wik, vji)
		
		if flag1 * flag2 < 0 and flag3 * flag4 < 0: return True
		return False
	
	def mouse_drag_left(self, event):
		self.label_e["text"] = str(event)
		ratio = math.pow(self.MAGNIFICATION, self.level)
		delta_x = (event.x - self.left_pos[0]) / ratio
		delta_y = (event.y - self.left_pos[1]) / ratio
		self.left_pos[:] = event.x, event.y
		if not self.is_edge_locked and self.corner_drag_check(delta_x, delta_y):
			corners_dst = np.copy(self.corners)
			corners_dst[self.targ_id] += delta_x, delta_y
			self.update_nodes(corners_dst)
		else:
			i = self.targ_id // (self.grid_hn + 1)
			j = self.targ_id % (self.grid_hn + 1)
			if 0 < i < self.grid_wn and 0 < j < self.grid_hn:
				x0, y0 = self.node_map[i][j].get_pos()
				self.node_map[i][j].set_pos(x0 + delta_x, y0 + delta_y)
				pass
			pass
		
		self.update()
	
	def is_line_in_display(self, x1, y1, x2, y2):
		if x1 < 0 and x2 < 0: return False
		if y1 < 0 and y2 < 0: return False
		if x1 > self.can_width and x2 > self.can_width: return False
		if y1 > self.can_height and y2 > self.can_height: return False
		return True
	
	def draw_grid(self):
		ratio = math.pow(self.MAGNIFICATION, self.level)
		
		self.tk_grid = []
		for i in range(1, self.grid_wn):
			x0, y0 = self.node_map[i][0].get_pos()
			x0 = round(x0 * ratio + self.loc_offset[0])
			y0 = round(y0 * ratio + self.loc_offset[1])
			row_pts = [(x0, y0)]
			for j in range(1, self.grid_hn + 1):
				x1, y1 = self.node_map[i][j].get_pos()
				x1 = round(x1 * ratio + self.loc_offset[0])
				y1 = round(y1 * ratio + self.loc_offset[1])
				if self.is_line_in_display(x0, y0, x1, y1):
					row_pts.append((x1, y1))
				elif len(row_pts) == 1:
					row_pts = [(x1, y1)]
				else: break
				x0, y0 = x1, y1
			if len(row_pts) > 1:
				self.tk_grid.append(self.create_line(row_pts, width = 1, fill = "white"))
				self.tk_grid.append(self.create_line(row_pts, width = 1, fill = "black", dash = [5, 5]))
				pass
			pass
		for j in range(1, self.grid_hn):
			x0, y0 = self.node_map[0][j].get_pos()
			x0 = round(x0 * ratio + self.loc_offset[0])
			y0 = round(y0 * ratio + self.loc_offset[1])
			col_pts = [(x0, y0)]
			for i in range(1, self.grid_wn + 1):
				x1, y1 = self.node_map[i][j].get_pos()
				x1 = round(x1 * ratio + self.loc_offset[0])
				y1 = round(y1 * ratio + self.loc_offset[1])
				if self.is_line_in_display(x0, y0, x1, y1):
					col_pts.append((x1, y1))
				elif len(col_pts) == 1:
					col_pts = [(x1, y1)]
				else: break
				x0, y0 = x1, y1
			if len(col_pts) > 1:
				self.tk_grid.append(self.create_line(col_pts, width = 1, fill = "white"))
				self.tk_grid.append(self.create_line(col_pts, width = 1, fill = "black", dash = [5, 5]))
				pass
			pass
		pass
	
	def draw_rect(self):
		ratio = math.pow(self.MAGNIFICATION, self.level)
		loc_corners = self.corners * ratio + self.loc_offset
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
	
	def draw_line(self):
		if self.left_press:
			x1, y1 = self.left_pos
			ratio = math.pow(self.MAGNIFICATION, self.level)
			if not self.is_edge_locked:
				x0, y0 = self.corners[self.targ_id] * ratio + self.loc_offset
				x0 = float((x0 - x1) * 0.9 + x1)
				y0 = float((y0 - y1) * 0.9 + y1)
			else:
				i = self.targ_id // (self.grid_hn + 1)
				j = self.targ_id % (self.grid_hn + 1)
				x0, y0 = np.asarray(self.node_map[i][j].get_pos()) * ratio + self.loc_offset
				x0 = float((x0 - x1) * 0.8 + x1)
				y0 = float((y0 - y1) * 0.8 + y1)
			self.tk_line = self.create_line(x0, y0, x1, y1, width = 1, fill = "plum1", dash = [12, 4, 4, 4])
	
	def update(self):
		self.tk_delete()
		
		self.draw_image()
		
		self.draw_grid()
		
		self.draw_line()
		
		self.draw_rect()
	
	def node_cross_detect(self):
		w = round(self.grid_wa * 2 * self.PPMM)
		h = round(self.grid_ha * 2 * self.PPMM)
		mask = circle_mask(w, h, 0.90)
		# src = self.img_cash[0][2]
		src = cv2.cvtColor(self.img_cash[0][2], cv2.COLOR_BGR2GRAY)
		src = cv2.GaussianBlur(src, (3, 3), 1.414)
		
		center_pts = np.zeros((self.grid_wn + 1, self.grid_hn + 1, 2))
		flags = np.full((self.grid_wn + 1, self.grid_hn + 1), False)
		for t in range(50):
			is_changed = False
			print(f"==================== round {t} ====================")
			for i in range(1, self.grid_wn):
				for j in range(1, self.grid_hn):
					if flags[i][j]: continue
					flags[i][j] = True
					center_pts[i, j] = self.node_map[i][j].detect(w, h, src, mask, self.DETECT_THRESHOLD)
					print(i, j, center_pts[i, j].round(2))
					pass
				pass
			for i in range(1, self.grid_wn):
				for j in range(1, self.grid_hn):
					x0, y0 = self.node_map[i][j].get_pos()
					x1, y1 = center_pts[i, j]
					if (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) > 0.2:
						flags[i][j] = False
						flags[i - 1][j - 1] = False
						flags[i - 1][j + 1] = False
						flags[i + 1][j + 1] = False
						flags[i + 1][j - 1] = False
						is_changed = True
					self.node_map[i][j].set_pos(x1, y1)
					pass
				pass
			if not is_changed: break
		self.update()
		pass
	
	def get_crop(self):
		w = round(self.grid_wn * self.grid_wa * self.PPMM)
		h = round(self.grid_hn * self.grid_ha * self.PPMM)
		corner_dst = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
		mat = cv2.getPerspectiveTransform(self.corners, corner_dst)
		src = self.img_cash[0][2]
		dst = cv2.warpPerspective(src, mat, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT)
		return dst
	
	def delete_grid(self):
		if self.tk_grid is not None:
			for tl in self.tk_grid:
				self.delete(tl)
	
	def tk_delete(self):
		self.delete(self.tk_line)
		self.delete(self.tk_rect)
		self.delete(self.tk_ellipse)
		self.delete_grid()


if __name__ == '__main__':
	import tkinter as tk
	
	sys.path.append(os.path.dirname(__file__))
	
	folder = "/home/lab/Desktop/python_resource/M24_12/D2412_28/out/"
	
	WIDTH = 1200
	HEIGHT = 1000
	
	
	def main():
		win = tk.Tk()
		win.title("test display canvas")
		win.geometry(f"{WIDTH}x{HEIGHT}+8+8")
		win.resizable(False, False)
		
		can = MainCanvas(win, WIDTH, HEIGHT)
		can.place(x = 0, y = 0)
		
		path = "/home/lab/Desktop/python_resource/M24_12/D2412_20/res/img_4.png"
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		can.init_image(img)
		can.init_grid_info(58, 40, 9.99, 9.99)
		# can.set_nodes()
		# can.is_edge_locked = True
		# can.node_cross_detect()
		
		# dst = can.get_crop()
		# cv2.imwrite(folder + "test.jpg", dst)
		
		win.mainloop()
	
	
	main()
	pass
