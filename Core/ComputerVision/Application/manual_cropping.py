import math
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from fontTools.cffLib.specializer import commandsToProgram

if __name__ == "__main__":
	FOLDER = "/home/lab/Desktop/python_resource/M24_12/D2412_20/res/"
	UPDATE_share = False
	
	
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
			
			super().__init__(win, relief = tk.RAISED, width = width, height = height)
			
			# 鼠标事件报告
			self.label_e = tk.Label(self, text = "Helo event .. ", foreground = "#0000ff")
			self.label_e.place(x = 5, y = 5)
		
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
		
		def mouse_press_middle(self, event):
			self.label_e["text"] = str(event)
			self.middle_press = True
			self.middle_pos[:] = event.x, event.y
		
		def mouse_press_right(self, event):
			self.label_e["text"] = str(event)
			self.right_press = True
			self.right_pos[:] = event.x, event.y
		
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
			self.label_e["text"] = str(event)
		
		def mouse_double_middle(self, event):
			self.label_e["text"] = str(event)
		
		def mouse_double_right(self, event):
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
		
		def mouse_press_left(self, event):
			super().mouse_press_left(event)
			
			loc_corners = self.get_loc_corners()
			diff = loc_corners - (event.x, event.y)
			d = np.sum(diff * diff, axis = 1)
			d[4:] *= 3
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
			super().update()
			
			loc_corners = self.get_loc_corners()
			if self.left_press:
				x0, y0 = loc_corners[self.corner_id]
				x1, y1 = self.left_pos
				x0 = self.pos_calc(x0, x1, 0.95)
				y0 = self.pos_calc(y0, y1, 0.95)
				self.tk_line = self.create_line(x0, y0, x1, y1, width = 1, fill = "plum1", dash = [12, 4, 4, 4])
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
			loc_corners = self.get_loc_corners()
			
			corner_src = loc_corners[:4].astype(np.float32)
			corner_dst = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
			mat = cv2.getPerspectiveTransform(corner_dst, corner_src)
			
			self.tk_grid = []
			for i in range(1, a):
				den = float(mat[2, 0] * i / a + mat[2, 2])
				x0 = float((mat[0, 0] * i / a + mat[0, 2]) / den)
				y0 = float((mat[1, 0] * i / a + mat[1, 2]) / den)
				
				den = float(mat[2, 0] * i / a + mat[2, 1] + mat[2, 2])
				x1 = float((mat[0, 0] * i / a + mat[0, 1] + mat[0, 2]) / den)
				y1 = float((mat[1, 0] * i / a + mat[1, 1] + mat[1, 2]) / den)
				
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "white"))
				self.tk_grid.append(self.create_line(x0, y0, x1, y1, width = 1, fill = "black", dash = [5, 5]))
			for i in range(1, b):
				den = float(mat[2, 1] * i / b + mat[2, 2])
				x0 = float((mat[0, 1] * i / b + mat[0, 2]) / den)
				y0 = float((mat[1, 1] * i / b + mat[1, 2]) / den)
				
				den = float(mat[2, 0] + mat[2, 1] * i / b + mat[2, 2])
				x1 = float((mat[0, 0] + mat[0, 1] * i / b + mat[0, 2]) / den)
				y1 = float((mat[1, 0] + mat[1, 1] * i / b + mat[1, 2]) / den)
				
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
	
	
	class Application:
		WIN1_WIDTH = 1200
		WIN1_HEIGHT = 900
		WIN2_WIDTH = 640
		WIN2_HEIGHT = 640
		
		def __init__(self):
			self.working_folder = None
			self.working_name = None
			
			self.win = tk.Tk()
			self.win.title("Project")
			win_w = self.WIN1_WIDTH + self.WIN2_WIDTH
			win_h = self.WIN1_HEIGHT
			self.win.geometry(f"{win_w}x{win_h}+10+10")
			self.win.resizable(False, False)
			
			self.frame_1 = tk.Frame(self.win, relief = tk.RAISED, width = self.WIN1_WIDTH, height = self.WIN1_HEIGHT)
			self.frame_1.grid(row = 0, column = 0, sticky = tk.NW)
			
			self.frame_2 = tk.Frame(self.win, relief = tk.RAISED, width = self.WIN2_WIDTH, height = self.WIN1_HEIGHT)
			self.frame_2.grid(row = 0, column = 1, sticky = tk.NW)
			
			tk.Label(self.frame_2, text = "宽度/mm").grid(row = 0, column = 0)
			self.W_entry = tk.Entry(self.frame_2, width = 20, justify = "left")
			self.W_entry.grid(row = 0, column = 1, sticky = tk.W)
			self.W_entry.insert(tk.END, "100")
			
			tk.Label(self.frame_2, text = "高度/mm").grid(row = 1, column = 0)
			self.H_entry = tk.Entry(self.frame_2, width = 20, justify = "left")
			self.H_entry.grid(row = 1, column = 1, sticky = tk.W)
			self.H_entry.insert(tk.END, "100")
			
			self.W_entry.bind("<ButtonPress-1>", lambda e: (
				self.W_entry.config(state = "normal"),
				self.H_entry.config(state = "normal"),
			))
			self.H_entry.bind("<ButtonPress-1>", lambda e: (
				self.H_entry.config(state = "normal"),
				self.W_entry.config(state = "normal"),
			))
			
			B_import = tk.Button(self.frame_2, text = "input", command = self.get_image_path)
			B_import.grid(row = 2, column = 0)
			B_output = tk.Button(self.frame_2, text = "save", command = self.save_image)
			B_output.grid(row = 2, column = 1)
			B_update = tk.Button(self.frame_2, text = "update", command = self.update)
			B_update.grid(row = 2, column = 2)
			
			self.win.bind("<Return>", lambda e: self.update())
			
			self.can_1 = MainCanvas(self.frame_1, self.WIN1_WIDTH, self.WIN1_HEIGHT)
			self.can_1.place(x = 0, y = 0)
			
			self.can_2 = DisplayCanvas(self.frame_2, self.WIN2_WIDTH, self.WIN2_HEIGHT)
			self.can_2.grid(row = 3, column = 0, columnspan = 40)
			
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
					self.can_1.init_image(img)
			except FileNotFoundError:
				return
		
		def get_image_real_size(self):
			w = self.W_entry.get()
			h = self.H_entry.get()
			try:
				w = float(w)
			except ValueError:
				self.W_entry.delete(0, tk.END)
				self.W_entry.insert(tk.END, "100")
				w = 100
			try:
				h = float(h)
			except ValueError:
				self.H_entry.delete(0, tk.END)
				self.H_entry.insert(tk.END, "100")
				h = 100
			return w, h
		
		def save_image(self):
			w_real, h_real = self.get_image_real_size()
			img = self.can_1.get_crop(w_real, h_real)
			
			path = self.working_folder + "warped_" + self.working_name
			cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		
		def run(self):
			self.win.mainloop()
		
		def update(self):
			self.W_entry.config(state = "readonly")
			self.H_entry.config(state = "readonly")
			if self.can_1.is_init is True:
				w, h = self.get_image_real_size()
				img = self.can_1.get_crop(w, h)
				self.can_2.init_image(img)
				self.can_2.update()
	
	
	def main():
		app = Application()
		app.run()
	
	
	main()
	
	pass
