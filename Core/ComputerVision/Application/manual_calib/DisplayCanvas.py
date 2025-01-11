import math
import tkinter as tk

import cv2
from PIL import Image, ImageTk


class DisplayCanvas(tk.Canvas):
	MAGNIFICATION = math.sqrt(2)
	MIN_PIXEL = 200  # zoom out min pixel
	MAX_PIXEL = 24000  # zoom in max pixel
	
	def __init__(self, win, width, height):
		self.is_init = False
		self.can_width = width
		self.can_height = height
		
		# interface parameter
		self.left_pos = [0, 0]
		self.right_pos = [0, 0]
		self.middle_pos = [0, 0]
		self.left_press = False
		self.right_press = False
		self.middle_press = False
		
		# display image parameter
		self.level = 0
		self.min_level = 0
		self.max_level = 0
		self.loc_img = None
		self.loc_size = [0, 0]
		self.loc_offset = [0, 0]
		self.img_cash = None
		self.tk_img = None
		
		super().__init__(win, relief = tk.FLAT, width = width, height = height)
		
		# mouse event reporter
		self.label_e = tk.Label(self, text = "Helo event .. ")
		self.label_e.place(x = 8, y = 8)
	
	def set_level(self, level):
		self.level = level
		w, h, img = self.img_cash[level]
		self.loc_size[:] = w, h
		self.loc_img = img
		pass
	
	def init_image(self, image):
		self.is_init = True
		
		h, w = image.shape[:2]
		
		self.img_cash = dict()
		self.img_cash = {0: (w, h, image)}
		
		tmp_w = w
		self.min_level = 0
		while tmp_w / self.MAGNIFICATION > self.MIN_PIXEL:
			new_level = self.min_level - 1
			ratio = math.pow(self.MAGNIFICATION, new_level)
			tmp_w = round(w * ratio)
			tmp_h = round(h * ratio)
			img = self.img_cash[self.min_level][2]
			img = cv2.resize(img, (tmp_w, tmp_h), interpolation = cv2.INTER_LINEAR)
			self.img_cash[new_level] = (tmp_w, tmp_h, img)
			self.min_level = new_level
		
		tmp_w = w
		self.max_level = 0
		while tmp_w * self.MAGNIFICATION < self.MAX_PIXEL:
			new_level = self.max_level + 1
			ratio = math.pow(self.MAGNIFICATION, new_level)
			tmp_w = round(w * ratio)
			tmp_h = round(h * ratio)
			img = self.img_cash[self.max_level][2]
			img = cv2.resize(img, (tmp_w, tmp_h), interpolation = cv2.INTER_LINEAR)
			self.img_cash[new_level] = (tmp_w, tmp_h, img)
			self.max_level = new_level
		
		self.set_level(self.level)
		
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
	
	pass
	
	# 鼠标进入
	def mouse_enter(self, event):
		self.label_e["text"] = str(event)
	
	# 鼠标离开
	def mouse_leave(self, event):
		self.label_e["text"] = str(event)
	
	# 鼠标移动
	def mouse_move(self, event):
		self.label_e["text"] = str(event)
	
	# 鼠标滚轮
	def mouse_wheel_up(self, event):
		self.label_e["text"] = str(event)
		if self.level < self.max_level:
			self.set_level(self.level + 1)
			cx = self.can_width / 2
			cy = self.can_height / 2
			k = self.MAGNIFICATION
			self.loc_offset[:] = [
				round((self.loc_offset[0] - cx) * k + cx),
				round((self.loc_offset[1] - cy) * k + cy),
			]
			self.update()
	
	def mouse_wheel_down(self, event):
		self.label_e["text"] = str(event)
		if self.level > self.min_level:
			self.set_level(self.level - 1)
			cx = self.can_width / 2
			cy = self.can_height / 2
			k = 1 / self.MAGNIFICATION
			self.loc_offset[:] = [
				round((self.loc_offset[0] - cx) * k + cx),
				round((self.loc_offset[1] - cy) * k + cy),
			]
			self.update()
	
	# 鼠标按下
	def mouse_press_left(self, event):
		self.focus_set()
		self.label_e["text"] = str(event)
		self.left_press = True
		self.left_pos[:] = event.x, event.y
	
	def mouse_press_middle(self, event):
		self.focus_set()
		self.label_e["text"] = str(event)
		self.middle_press = True
		self.middle_pos[:] = event.x, event.y
	
	def mouse_press_right(self, event):
		self.focus_set()
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
		self.loc_offset[0] += event.x - self.middle_pos[0]
		self.loc_offset[1] += event.y - self.middle_pos[1]
		self.middle_pos[:] = event.x, event.y
		
		self.update()
	
	def mouse_drag_right(self, event):
		self.label_e["text"] = str(event)
	
	def draw_image(self):
		x0, y0 = self.loc_offset
		x1, y1 = x0 + self.loc_size[0], y0 + self.loc_size[1]
		
		if x0 >= self.can_width: return
		if y0 >= self.can_height: return
		if x1 <= 0: return
		if y1 <= 0: return
		
		if x0 < 0: w0, a = -x0, 0
		else: w0, a = 0, x0
		if y0 < 0: h0, b = -y0, 0
		else: h0, b = 0, y0
		
		if x1 < self.can_width: w1 = self.loc_size[0]
		else: w1 = self.can_width - x0
		if y1 < self.can_height: h1 = self.loc_size[1]
		else: h1 = self.can_height - y0
		
		img = self.loc_img[h0:h1, w0:w1, :]
		self.tk_img = ImageTk.PhotoImage(Image.fromarray(img))
		self.create_image(a, b, image = self.tk_img, anchor = "nw")
	
	def update(self):
		self.draw_image()
		pass


if __name__ == '__main__':
	WIDTH = 600
	HEIGHT = 400
	
	
	def main():
		win = tk.Tk()
		win.title("test display canvas")
		win.geometry(f"{WIDTH}x{HEIGHT}+8+8")
		win.resizable(False, False)
		
		can = DisplayCanvas(win, WIDTH, HEIGHT)
		can.place(x = 0, y = 0)
		
		path = "/home/lab/Desktop/python_resource/M24_12/D2412_20/res/img_4.png"
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		can.init_image(img)
		
		win.mainloop()
	
	
	main()
