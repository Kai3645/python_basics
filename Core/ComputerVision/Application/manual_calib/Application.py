import math
import os
import re
import sys

import cv2
import numpy as np

if __name__ == "__main__":
	import tkinter as tk
	from tkinter import filedialog
	
	from MainCanvas import MainCanvas
	
	sys.path.append(os.path.dirname(__file__))
	
	FOLDER = "/home/lab/Desktop/python_resource/M25_01/D2501_03/out/"
	
	
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
			
			B_equal = tk.Button(self.frame_2, text = "等边", width = 2, command = copy_upper)
			B_equal.grid(row = 0, column = 4, sticky = tk.S)
			self.is_lock_entry = False
			
			def confirm_entry():
				if not self.is_lock_entry and self.can.is_init:
					try:
						wn = int(self.E_w_n.get())
						hn = int(self.E_h_n.get())
						wa = float(self.E_w_a.get())
						ha = float(self.E_h_a.get())
						
						self.is_lock_entry = True
						B_confirm.config(text = "取消")
						self.E_w_n.config(state = "readonly")
						self.E_w_a.config(state = "readonly")
						self.E_h_n.config(state = "readonly")
						self.E_h_a.config(state = "readonly")
						self.can.init_grid_info(wn, hn, wa, ha)
						return
					except ValueError:
						pass
				self.is_lock_entry = False
				B_confirm.config(text = "确认")
				self.E_w_n.config(state = "normal")
				self.E_w_a.config(state = "normal")
				self.E_h_n.config(state = "normal")
				self.E_h_a.config(state = "normal")
				pass
			
			B_confirm = tk.Button(self.frame_2, text = "确认", width = 2, command = confirm_entry)
			B_confirm.grid(row = 1, column = 4, sticky = tk.S)
			
			loc_frame = tk.Frame(self.frame_2, relief = tk.FLAT)
			loc_frame.grid(row = 2, column = 0, columnspan = 4, sticky = tk.W)
			loc_frame.grid_rowconfigure(0, minsize = 50)
			
			self.is_detected = False
			
			def get_image_path():
				path = filedialog.askopenfilename(
					filetypes = [("image", r"*.jpeg  *.jpg *.png")],
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
						self.is_detected = False
						self.calibrate_flag = True
				except FileNotFoundError:
					pass
			
			def lock_edge():
				if not self.is_lock_entry:
					print("entry data not ready yet ..")
					return
				if B_lock["text"] == "锁边":
					B_lock.config(text = "解锁")
					self.can.is_edge_locked = True
				elif B_lock["text"] == "解锁":
					B_lock.config(text = "锁边")
					self.can.is_edge_locked = False
			
			def detection():
				if not self.is_lock_entry:
					print("entry data not initialization yet ..")
					return
				else:
					self.can.node_cross_detect()
					self.is_detected = True
			
			self.pts_obj_cash = []
			self.pts_img_cash = []
			self.loop_count = 0
			
			self.calibrate_cash = None
			self.calibrate_flag = True
			
			def update_calibration():
				if self.can.is_init and self.is_detected and self.calibrate_flag:
					pts_img = []
					pts_obj = []
					for i in range(1, self.can.grid_wn):
						for j in range(1, self.can.grid_hn):
							x1, y1 = self.can.node_map[i][j].get_pos()
							pts_img.append([x1, y1])
							x2 = self.can.grid_wa * i
							y2 = self.can.grid_ha * j
							pts_obj.append([x2, y2, 0])
					if len(pts_img) < 20: return
					
					pts_obj = np.asarray(pts_obj, np.float32)
					pts_img = np.asarray(pts_img, np.float32)
					print("pts_obj shape = ", pts_obj.shape)
					print("pts_img shape = ", pts_img.shape)
					print()
					self.pts_obj_cash.append(pts_obj)
					self.pts_img_cash.append(pts_img)
					
					w, h, src = self.can.img_cash[0]
					mat = np.asarray([[h, 0, w / 2], [0, h, h / 2], [0, 0, 1]])
					dist = np.zeros(14)
					self.calibrate_cash = mse_err, mat, dist, Rs, Ts = cv2.calibrateCamera(
						self.pts_obj_cash, self.pts_img_cash, (w, h), mat, dist,
						flags = (  # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty
							cv2.CALIB_RATIONAL_MODEL |  # using k4, k5, k6, upgrade to len(dist) == 8
							cv2.CALIB_THIN_PRISM_MODEL |  # using s1, s2, s3, s4, upgrade to len(dist) == 12
							cv2.CALIB_TILTED_MODEL |  # using tauX, tauY, upgrade to len(dist) == 14
							cv2.CALIB_USE_INTRINSIC_GUESS |  # give init fx, fy, cx, cy, or (cx = 0.5 * W, cy = 0.5 * H)
							# cv2.CALIB_USE_INTRINSIC_GUESS |  # cx = 0.5 * W, cy = 0.5 * H
							cv2.CALIB_FIX_ASPECT_RATIO |  # const fx / fy
							# cv2.CALIB_FIX_FOCAL_LENGTH |  # const fx, fy
							# cv2.CALIB_ZERO_TANGENT_DIST |  # const p1, p2 == 0
							# cv2.CALIB_FIX_K1 |  # const Ki == 0, or init by CALIB_USE_INTRINSIC_GUESS
							# cv2.CALIB_FIX_S1_S2_S3_S4 |  # const s1, s2, s3, s4 == 0, or init by CALIB_USE_INTRINSIC_GUESS
							# cv2.CALIB_FIX_TAUX_TAUY |  # const tauX, tauY == 0, or init by CALIB_USE_INTRINSIC_GUESS
							0  # for format only
						)
					)
					T_info.config(state = "normal")
					T_info.insert(tk.END, f"------------------------------\n"
					                      f"Calibrate Loop No.{self.loop_count} \n"
					                      f"  mse_err = {mse_err:.5f}\n"
					                      f"  fx = {mat[0, 0]:.2f}\n"
					                      f"  fy = {mat[1, 1]:.2f}\n"
					                      f"  cx = {mat[0, 2]:.2f}\n"
					                      f"  cy = {mat[1, 2]:.2f}\n"
					                      f"  dist = \n"
					                      f"    k1 >> {dist[0]:.8f}\n"
					                      f"    k2 >> {dist[1]:.8f}\n"
					                      f"    p1 >> {dist[2]:.8f}\n"
					                      f"    p2 >> {dist[3]:.8f}\n"
					                      f"    k3 >> {dist[4]:.8f}\n"
					                      f"    k4 >> {dist[5]:.8f}\n"
					                      f"    k5 >> {dist[6]:.8f}\n"
					                      f"    k6 >> {dist[7]:.8f}\n"
					                      f"    s1 >> {dist[8]:.8f}\n"
					                      f"    s2 >> {dist[9]:.8f}\n"
					                      f"    s3 >> {dist[10]:.8f}\n"
					                      f"    s4 >> {dist[11]:.8f}\n"
					                      f"    tx >> {dist[12]:.8f}\n"
					                      f"    ty >> {dist[13]:.8f}\n"
					                      f"==============================\n")
					T_info.config(state = "disabled")
					T_info.see(tk.END)
					
					print("mse_err = ", mse_err)
					print("mat = ", mat)
					print("dist = ", dist)
					print()
					
					# undistort
					new_mat, roi = cv2.getOptimalNewCameraMatrix(mat, dist, (w, h), 1)
					# x, y, w, h = roi
					dst = cv2.undistort(src, mat, dist, None, new_mat)
					path = self.working_folder + f"undist_{self.loop_count:02d}_" + self.working_name
					cv2.imwrite(path, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
					
					_, rvec, tvec = cv2.solvePnP(pts_obj, pts_img, mat, dist, flags = cv2.SOLVEPNP_IPPE)
					reproj, _ = cv2.projectPoints(pts_obj, rvec, tvec, mat, dist)
					reproj = reproj.reshape((-1, 2))
					err = np.linalg.norm(reproj - pts_img, axis = 1)
					mse_errs = math.sqrt(np.dot(err, err) / (len(reproj) - 1))
					
					self.loop_count += 1
					print("manually calculated mse_errs = ", mse_errs)
					print()
					
					self.can.init_image(dst)
					self.is_detected = False
					self.calibrate_flag = False
				else: print("calibration canceled ..")
			
			def undistort():
				if self.calibrate_cash is None: return
				
				w, h, src = self.can.img_cash[0]
				mse_err, mat, dist, Rs, Ts = self.calibrate_cash
				
				new_mat, roi = cv2.getOptimalNewCameraMatrix(mat, dist, (w, h), 1)
				dst = cv2.undistort(src, mat, dist, None, new_mat)
				path = self.working_folder + f"undist_{self.loop_count:02d}_" + self.working_name
				cv2.imwrite(path, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
				
				self.can.init_image(dst)
				self.is_detected = False
				self.calibrate_flag = False
				pass
			
			def cropping():
				img = self.can.get_crop()
				path = self.working_folder + "warped_" + self.working_name
				cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
			
			B_import = tk.Button(loc_frame, text = "加载", command = get_image_path, justify = tk.CENTER)
			B_import.grid(row = 0, column = 0, sticky = tk.SE)
			loc_frame.grid_columnconfigure(0, minsize = 60)
			
			B_lock = tk.Button(loc_frame, text = "锁边", command = lock_edge, justify = tk.CENTER)
			B_lock.grid(row = 0, column = 1, sticky = tk.SE)
			loc_frame.grid_columnconfigure(1, minsize = 60)
			
			B_detect = tk.Button(loc_frame, text = "吸附", command = detection, justify = tk.CENTER)
			B_detect.grid(row = 0, column = 2, sticky = tk.SE)
			loc_frame.grid_columnconfigure(2, minsize = 60)
			
			B_update = tk.Button(loc_frame, text = "校准", command = update_calibration, justify = tk.CENTER)
			B_update.grid(row = 0, column = 3, sticky = tk.SE)
			loc_frame.grid_columnconfigure(3, minsize = 60)
			
			B_undistort = tk.Button(loc_frame, text = "矫正", command = undistort, justify = tk.CENTER)
			B_undistort.grid(row = 0, column = 4, sticky = tk.SE)
			loc_frame.grid_columnconfigure(4, minsize = 60)
			
			B_undistort = tk.Button(loc_frame, text = "裁剪", command = cropping, justify = tk.CENTER)
			B_undistort.grid(row = 0, column = 5, sticky = tk.SE)
			loc_frame.grid_columnconfigure(5, minsize = 60)
			
			T_info = tk.Text(self.frame_2, width = 50, height = 48, wrap = "word", undo = True)
			T_info.grid(row = 3, column = 0, columnspan = 8, sticky = "nsew")
			T_info.config(state = "disabled")
			
			self.working_folder = ""
			self.working_name = ""
			
			pass
		
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
