import math

import cv2
import numpy as np

from Core.ComputerVision.Application.manual_calib.DisplayCanvas import DisplayCanvas
from Core.Statistic.fitting_2d import fit_line2d_PCA

if __name__ == "__main__":
	import tkinter as tk
	from tkinter import filedialog
	
	PPMM = 10
	WIDTH = 1200
	HEIGHT = 1000
	SUB_WIDTH = 256
	SUB_WIDTH_TXT = SUB_WIDTH // 8 - 1
	
	FOLDER = "/home/lab/Desktop/python_resource/M25_01/D2501_05/out/"
	
	
	def is_line_in_rect(x1, y1, x2, y2, W = WIDTH, H = HEIGHT):
		if x1 < 0 and x2 < 0: return False
		if y1 < 0 and y2 < 0: return False
		if x1 > W and x2 > W: return False
		if y1 > H and y2 > H: return False
		return True
	
	
	class MyObject:
		def __init__(self):
			self.pts = []
			c = np.random.random(3)
			while np.sum(c) < 1.2:
				c = np.random.random(3)
			c = (c * (250 - 35) + 35).round()
			self.color = (int(c[0]), int(c[1]), int(c[2]))
			self.tk_objs = []
			pass
		
		def add_point(self, x, y):
			self.pts.append([x, y])
			self.calc_fitting()
		
		def get_roi(self):
			if len(self.pts) == 0: return None
			PPMM5 = 5 * PPMM
			pts = np.asarray(self.pts)
			x0 = int(np.min(pts[:, 0])) - PPMM5
			y0 = int(np.min(pts[:, 1])) - PPMM5
			x1 = math.ceil(np.max(pts[:, 0])) + PPMM5 + 1
			y1 = math.ceil(np.max(pts[:, 1])) + PPMM5 + 1
			return x0, x1, y0, y1
		
		def calc_fitting(self):
			if len(self.pts) < 3: return None
			pass
		
		def tk_draw_fitting(self, canvas, ratio, offset):
			pass
		
		def tk_draw_points(self, canvas, ratio, offset):
			loc_pts = np.array(self.pts) * ratio + offset
			for pt in loc_pts:
				x0, y0 = pt
				if x0 < 0 or x0 > WIDTH: continue
				if y0 < 0 or y0 > HEIGHT: continue
				x1, y1 = pt - 20
				x2, y2 = pt + 20
				self.tk_objs.append(canvas.create_line(
					round(x0), round(y1), round(x0), round(y2),
					fill = self.color, width = 1,
				))
				self.tk_objs.append(canvas.create_line(
					round(x1), round(y0), round(x2), round(y0),
					fill = self.color, width = 1,
				))
			pass
		
		def tk_draw(self, canvas, ratio, offset):
			if len(self.pts) > 3: self.tk_draw_fitting(canvas, ratio, offset)
			if len(self.pts) > 0: self.tk_draw_points(canvas, ratio, offset)
			pass
		
		def tk_delete(self, canvas):
			for obj in self.tk_objs:
				canvas.delete(obj)
				pass
			pass
	
	
	class MyLine(MyObject):
		def __init__(self):
			MyObject.__init__(self)
			
			self.p = None
			self.v = None
			self.th = PPMM
			pass
		
		def calc_fitting(self):
			if len(self.pts) < 3:
				self.p = None
				self.v = None
				return
			nx, ny, d = fit_line2d_PCA(self.pts)
			Ds = np.abs(np.dot(self.pts, (nx, ny)))
			th = 2 * np.mean(Ds)
			valid = Ds < th
			if np.sum(valid) < 3:
				self.p = np.mean(self.pts, axis = 0)
				self.v = np.asarray((ny, -nx))
				return
			nx, ny, d = fit_line2d_PCA(self.pts[valid])
			self.p = np.mean(self.pts[valid], axis = 0)
			self.v = np.asarray((ny, -nx))
			pass
		
		def tk_draw_fitting(self, canvas, ratio, offset):
			ds = np.dot(self.pts, self.v)
			d1 = np.max(ds) + 160
			d2 = np.min(ds) + 160
			p1 = ((self.p + d1 * self.v) * ratio + offset).round()
			p2 = ((self.p + d2 * self.v) * ratio + offset).round()
			x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
			if is_line_in_rect(x1, y1, x2, y2):
				self.tk_objs.append(canvas.create_line(x1, y1, x2, y2, fill = "white", width = 1))
				self.tk_objs.append(canvas.create_line(x1, y1, x2, y2, fill = "black", width = 1, dash = [5, 5]))
			delta = np.asarray((self.v[1], -self.v[0])) * self.th * ratio
			p1_ = (p1 + delta).round()
			p2_ = (p2 + delta).round()
			x1, y1, x2, y2 = int(p1_[0]), int(p1_[1]), int(p2_[0]), int(p2_[1])
			if is_line_in_rect(x1, y1, x2, y2):
				self.tk_objs.append(canvas.create_line(x1, y1, x2, y2, fill = "pink", width = 1, dash = [6, 6]))
			p1_ = (p1 - delta).round()
			p2_ = (p2 - delta).round()
			x1, y1, x2, y2 = int(p1_[0]), int(p1_[1]), int(p2_[0]), int(p2_[1])
			if is_line_in_rect(x1, y1, x2, y2):
				self.tk_objs.append(canvas.create_line(x1, y1, x2, y2, fill = "pink", width = 1, dash = [6, 6]))
			pass
	
	
	class MyCircle(MyObject):
		def __init__(self):
			MyObject.__init__(self)
			
			self.p = None
			self.r = 0
			pass
		
		def tk_draw_fitting(self, canvas, ratio, offset):
			p0 = self.p * ratio + offset
			r = self.r * ratio
			p1 = (p0 - r).round()
			p2 = (p0 + r).round()
			self.tk_objs.append(canvas.create_oval(
				int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]),
				fill = "white", width = 1,
			))
			self.tk_objs.append(canvas.create_oval(
				int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]),
				fill = "black", width = 1, dash = [5, 5],
			))
			pass
	
	
	class Application:
		
		def __init__(self):
			win = tk.Tk()
			win.title("Edge Detection")
			win.geometry(f"{WIDTH + SUB_WIDTH}x{HEIGHT}+8+8")
			win.resizable(False, False)
			
			# Frame Left
			F1 = tk.Frame(win, relief = tk.FLAT, width = WIDTH, height = HEIGHT)
			F1.grid(row = 0, column = 0, sticky = tk.NW)
			
			can = DisplayCanvas(F1, WIDTH, HEIGHT)
			can.place(x = 0, y = 0)
			
			# Frame Right
			F2 = tk.Frame(win, relief = tk.FLAT, width = SUB_WIDTH, height = HEIGHT)
			F2.grid(row = 0, column = 1, sticky = tk.NW)
			
			self.item_count = 0
			self.obj_items = dict()
			self.current_key = None  # mark: self.current_obj can be None
			
			def list_select(event):
				try:
					print(F2_LB.curselection(), event)
					cur = F2_LB.curselection()[0]
					self.current_key = F2_LB.get(cur)
				except IndexError: pass
			
			F2_LB = tk.Listbox(F2, width = SUB_WIDTH_TXT, height = 18, justify = tk.LEFT)
			F2_LB.grid(row = 1, column = 0, sticky = "nsew")
			F2_LB.bind("<<ListboxSelect>>", list_select)
			
			F2_F = tk.Frame(F2, relief = tk.FLAT)
			F2_F.grid(row = 2, column = 0, sticky = tk.W)
			F2_F.grid_rowconfigure(0, minsize = 50)
			
			def load_image():
				path = filedialog.askopenfilename(
					filetypes = [("image", r"*.jpeg  *.jpg *.png")],
					initialdir = FOLDER,
				)
				if type(path) is not str: return
				if path == "": return
				try:
					img = cv2.imread(path, cv2.IMREAD_COLOR)
					if img is not None:
						img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
						can.init_image(img)
				except FileNotFoundError:
					pass
			
			B_import = tk.Button(F2_F, text = "加载", command = load_image,
			                     width = 2, justify = tk.CENTER)
			B_import.grid(row = 0, column = 0, sticky = tk.W)
			F2_F.grid_columnconfigure(0, minsize = 50)
			
			def list_insert(type_name: str, new_obj):
				name = f"{self.item_count:04d}_{type_name}"
				F2_LB.insert('end', name)
				self.current_key = name
				self.obj_items[name] = new_obj
				pass
			
			def list_insert_circle():
				new_obj = MyCircle()
				list_insert("circle", new_obj)
				self.item_count += 1
				pass
			
			def list_insert_line():
				new_obj = MyLine()
				list_insert("line", new_obj)
				self.item_count += 1
				pass
			
			B_insert_circle = tk.Button(F2_F, text = "新圆", command = list_insert_circle,
			                            width = 2, justify = tk.CENTER)
			B_insert_circle.grid(row = 0, column = 1, sticky = tk.W)
			F2_F.grid_columnconfigure(1, minsize = 50)
			
			B_insert_line = tk.Button(F2_F, text = "新线", command = list_insert_line,
			                          width = 2, justify = tk.CENTER)
			B_insert_line.grid(row = 0, column = 2, sticky = tk.W)
			F2_F.grid_columnconfigure(2, minsize = 50)
			
			def list_delete():
				try:
					print(F2_LB.curselection(), "delete")
					cur = F2_LB.curselection()[0]
					key_name = F2_LB.get(cur)
					F2_LB.delete(cur)
					del self.obj_items[key_name]
					self.current_key = None
				except IndexError: pass
			
			B_insert_line = tk.Button(F2_F, text = "删除", command = list_delete, width = 2, justify = tk.CENTER)
			B_insert_line.grid(row = 0, column = 3, sticky = tk.W)
			F2_F.grid_columnconfigure(3, minsize = 50)
			
			def save_all():
				# todo:
				pass
			
			B_insert_line = tk.Button(F2_F, text = "保存", command = save_all, width = 2, justify = tk.CENTER)
			B_insert_line.grid(row = 0, column = 4, sticky = tk.W)
			F2_F.grid_columnconfigure(4, minsize = 50)
			
			F2_T_info = tk.Text(F2, width = SUB_WIDTH_TXT, height = 34, wrap = "word", undo = True)
			F2_T_info.grid(row = 3, column = 0, sticky = "nsew")
			F2_T_info.config(state = "disabled")
			
			win.mainloop()
	
	
	def main():
		app = Application()
	
	
	main()
	
	pass
