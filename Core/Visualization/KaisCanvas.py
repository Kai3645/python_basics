import math

# from tkinter import TclError

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.interpolate
import scipy.stats
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from Core.Basic import color_print
from Core.Statistic import histogram
from Core.Visualization.KaisColor import KaisColor


class KaisCanvas:
	def __init__(self, **kwargs):
		# https://matplotlib.org/3.5.2/tutorials/introductory/customizing.html#using-style-sheets
		# ============================================================
		#  style initiation
		# ============================================================
		# 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh',
		# 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn',
		# 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette',
		# 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper',
		# 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
		# 'seaborn-whitegrid', 'tableau-colorblind10'
		if kwargs.get("dark_mode", True): plt.style.use('dark_background')
		elif "style" in kwargs: plt.style.use(kwargs["style"])
		if "prop_cycle" in kwargs: rcParams["axes.prop_cycle"] = kwargs["prop_cycle"]
		rcParams["scatter.edgecolors"] = "None"
		rcParams["hist.bins"] = "auto"
		rcParams["legend.frameon"] = False
		rcParams["grid.alpha"] = 0.4
		line_width = kwargs.get("line_width", 1.6)
		rcParams["lines.linewidth"] = line_width
		rcParams["patch.linewidth"] = line_width
		rcParams["axes.linewidth"] = line_width
		font_size = kwargs.get("font_size", 12)
		rcParams["font.size"] = font_size
		rcParams["axes.labelsize"] = font_size
		rcParams["xtick.labelsize"] = font_size
		rcParams["ytick.labelsize"] = font_size
		label_size = kwargs.get("label_size", 13)
		rcParams["legend.fontsize"] = label_size
		rcParams["axes.titlesize"] = label_size
		# ============================================================
		#  fig initiation
		# ============================================================
		self.figsize = size = kwargs.get("fig_size", (11, 8))  # [inch]
		edge = kwargs.get("fig_edge", (0.87, 0.35, 0.4, 0.5))  # [inch] to each edge (L, R, U, D)
		self.rect = (  # percentage in square
			edge[0] / size[0],
			edge[3] / size[1],
			max(size[0] - edge[0] - edge[1], 0) / size[0],
			max(size[1] - edge[2] - edge[3], 0) / size[1],
		)
		self.ax_aspect = (self.rect[2] - self.rect[0]) / (self.rect[3] - self.rect[1])  # width / height
		self.fig = plt.figure(figsize = self.figsize)
		self.ax = self.fig.add_axes(self.rect)
		self.layer = 20  # left some space for manually setting
		plt.ion()  # enable interactive mode
	
	def layer_update(self, step: int = 1):
		num = self.layer
		self.layer += step
		return num
	
	def set_axis(self, **kwargs):
		if kwargs.get("sci_on", False): self.ax.ticklabel_format(
			useMathText = True, axis = "both",
			style = "sci", scilimits = (0, 0),
		)
		elif "decimal" in kwargs:
			fmt = "{x:." + str(kwargs["decimal"]) + "f}"
			self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
			self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
		
		if kwargs.get("equal_axis", True): self.ax.axis("equal")
		else: self.ax.axis("auto")
		self.ax.tick_params(which = "both", direction = 'in')
		if kwargs.get("legend_on", False):
			self.ax.legend(loc = kwargs.get("legend_loc", "upper right"))
		if kwargs.get("grid_on", True): self.ax.grid(zorder = 0)
		if "xticks" in kwargs: self.ax.set_xticks(kwargs["xticks"])
		if "yticks" in kwargs: self.ax.set_yticks(kwargs["yticks"])
		if "xlim" in kwargs: self.ax.set_xlim(kwargs["xlim"])
		if "ylim" in kwargs: self.ax.set_ylim(kwargs["ylim"])
		if "xlabel" in kwargs: self.ax.set_xlabel(kwargs["xlabel"])
		if "ylabel" in kwargs: self.ax.set_ylabel(kwargs["ylabel"])
		if "title" in kwargs: self.ax.set_title(kwargs["title"])
	
	def force_equal_xlim(self, x1, x2, y_center):
		self.ax.axis("auto")
		w_half = (x2 - x1) / 2
		h_half = w_half / self.ax_aspect
		y1 = y_center - h_half
		y2 = y_center + h_half
		self.ax.set_xlim(x1, x2)
		self.ax.set_ylim(y1, y2)
	
	def force_equal_ylim(self, y1, y2, x_center):
		self.ax.axis("auto")
		h_half = (y2 - y1) / 2
		w_half = h_half * self.ax_aspect
		x1 = x_center - w_half
		x2 = x_center + w_half
		self.ax.set_xlim(x1, x2)
		self.ax.set_ylim(y1, y2)
	
	def save(self, path: str, dpi: int = 250):
		self.fig.savefig(path, dpi = dpi)
	
	def clear(self):
		self.ax.cla()
	
	def show(self, auto = False, sleep = 1):
		self.fig.show()
		if not auto:
			color_print("yellow", "press enter continue ..")
			plt.pause(sleep)  # time for drawing
			input()
		else:
			plt.pause(sleep)  # time for drawing
		pass
	
	@staticmethod
	def close():
		# try:
		plt.ioff()
		plt.close()
		# except TclError: pass
		pass
	
	# draw funcs
	def draw_points(self, points, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
		:param points: (n, 2) array like
		:param para: kwargs for pyplot.scatter
		:return: None
		"""
		points = np.atleast_2d(points)
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		self.ax.scatter(points[:, 0], points[:, 1], **para)
	
	def draw_polyline(self, points, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
		:param points: (n, 2) array like
		:param para: kwargs for pyplot.plot
		:return: None
		"""
		points = np.atleast_2d(points)
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		self.ax.plot(points[:, 0], points[:, 1], **para)
	
	def draw_line(self, point1, point2, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
		:param point1: (n, 2) array like
		:param point2: (n, 2) array like
		:param para: kwargs for lines.Line2D
		:return: None
		"""
		point1 = np.atleast_2d(point1)
		point2 = np.atleast_2d(point2)
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		flag = True
		for (p1, p2) in zip(point1, point2):
			line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), **para)
			if flag:  # remove repeated paras
				if "label" in para.keys(): para.pop("label")
				flag = False
			self.ax.add_line(line)
		pass
	
	def draw_vecRay(self, point, vector, length: float = 1, both_side = True, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
		:param point: (x, y)
		:param vector: (vx, vy)
		:param length:
		:param both_side:
		:param para: kwargs for lines.Line2D
		:return:
		"""
		pt0 = np.asarray(point, np.float64)
		vec = np.asarray(vector, np.float64)
		vec = vec * length / np.linalg.norm(vec)
		pt1 = pt0 + vec
		if both_side: pt0 = pt0 - vec
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		line = Line2D((pt0[0], pt1[0]), (pt0[1], pt1[1]), **para)
		self.ax.add_line(line)
	
	def draw_angleRay(self, point, angle, length: float = 1, both_side = True, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
		:param point: (x, y)
		:param angle: [rad]
		:param length:
		:param both_side:
		:param para: kwargs for lines.Line2D
		:return:
		"""
		vector = [math.cos(angle), math.sin(angle)]
		self.draw_vecRay(point, vector, length, both_side, **para)
	
	def draw_ellipse(self, point, vector, a, b, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html
		:param point: (x, y)
		:param vector: width direction for axis[0]
		:param a: half width
		:param b: half height
		:param para: kwargs for patches.Ellipse
		:return:
		"""
		angle = np.rad2deg(math.atan2(vector[1], vector[0]))
		para["fill"] = para.get("fill", False)
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		ellipse = Ellipse(point, a * 2, b * 2, angle = angle, **para)
		self.ax.add_patch(ellipse)
	
	def draw_circle(self, x, y, r, n = 200, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
		:param x: center x
		:param y: center y
		:param r: radius
		:param n: points number
		:param para: kwargs for pyplot.plot
		:return:
		"""
		T = np.linspace(0.0, math.pi * 2, n)
		P = np.zeros((n, 2))
		P[:, 0] = np.cos(T) * r + x
		P[:, 1] = np.sin(T) * r + y
		self.draw_polyline(P, **para)
	
	def draw_covariance(self, mean, covar, coe: float = 3, **para):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html
		:param mean: (x, y)
		:param covar: (4, 4) array like
 		:param coe:
		:param para: kwargs for patches.Ellipse
		:return:
			sigma: sigma array
			W: base vectors
		"""
		mean = np.asarray(mean, np.float64)
		covar = np.asarray(covar, np.float64)
		
		lam, W = np.linalg.eig(covar)
		idxes = lam.argsort()[::-1]
		lam = lam[idxes]
		W = W[:, idxes]
		
		sigma = np.sqrt(lam)
		a, b = sigma * coe
		para["color"] = para.get("color", KaisColor.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		self.draw_ellipse(mean, W[:, 0], a, b, **para)
		return sigma, W
	
	# graph funcs
	def histogram(self, data, normalize = True, denoise_on: bool = False, fit_on: bool = True, fit_type: str = "norm",
	              *,
	              step_para: dict = None, fit_para: dict = None,
	              ):
		"""
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.step.html?highlight=step#matplotlib.pyplot.step
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
		replace plan
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html?highlight=hist#matplotlib.pyplot.hist
		:param data:
		:param normalize:
		:param denoise_on: use RANSAC
		:param fit_on: default True
		:param fit_type: ("norm", "beta", "gamma", "pareto", "rayleigh")
		:param step_para: kwargs for pyplot.step
		:param fit_para: kwargs for pyplot.plot
		:return:
		"""
		length = len(data)
		if length < 10: return None
		D = np.asarray(data)
		
		x0, y0 = histogram(D, normalize = normalize)
		Q3i = int(length / 4 * 3)
		dD = np.abs(D - np.mean(D))
		idxes = np.argsort(dD)
		idx_best = idxes[0]
		for i in range(20):
			dD = np.abs(D - np.mean(D[idxes[:Q3i]]))
			idxes = np.argsort(dD)
			if idxes[0] == idx_best: break
			idx_best = idxes[0]
		x1, y1 = histogram(D[idxes[:Q3i]], normalize = normalize)
		
		if step_para is None: step_para = dict()
		if denoise_on:
			step_para["label"] = step_para.get("label", "hist")
			step_para["zorder"] = step_para.get("zorder", self.layer_update(2))
			self.ax.step(x1, y1, where = "mid", **step_para)
			
			step_para["label"] += "_raw"
			step_para["color"] = "gray"
			step_para["alpha"] = 0.7
			step_para["zorder"] -= 1
			self.ax.step(x0, y0, where = "mid", **step_para)
		else:
			step_para["label"] = step_para.get("label", "hist")
			step_para["zorder"] = step_para.get("zorder", self.layer_update(2))
			self.ax.step(x0, y0, where = "mid", **step_para)
		
		dist = getattr(scipy.stats, fit_type)
		num = (x0[0] - x0[-1]) / (x1[0] - x1[-1]) * 100
		T = np.linspace(x0[0], x0[-1], num = int(num))
		param = dist.fit(D[idxes[:Q3i]])
		if fit_on:
			pdf_fitted = dist.pdf(T, *param[:-2], loc = param[-2], scale = param[-1])
			
			if fit_para is None: fit_para = dict()
			fit_para["label"] = fit_para.get("label", "fitting-" + fit_type)
			fit_para["zorder"] = fit_para.get("zorder", self.layer_update())
			self.ax.plot(T, pdf_fitted, **fit_para)
		return param
	
	def hist_image_gray(self, image):
		result = np.zeros(256, int)
		index, count = np.unique(image.ravel(), return_counts = True)
		result[index] = count
		self.ax.step(np.arange(256), result, zorder = self.layer_update())
		return result
	
	def hist_image(self, image):
		src = np.atleast_3d(image)
		_, _, c = src.shape
		colors = ("gold", "deepskyblue", "orchid")
		result = np.zeros((256, c), int)
		for i in range(c):
			index, count = np.unique(src[:, :, i].ravel(), return_counts = True)
			result[index, i] = count
			self.ax.step(np.arange(256), result[:, i], color = colors[i],
			             label = f"layer {i}", zorder = self.layer_update())
		return result
	
	pass


if __name__ == '__main__':
	folder = "/home/lab/Desktop/python_resource/M24_10/D2410_23/out/"
	can = KaisCanvas(dark_mode = True, fig_size = (6, 4))
	
	
	def main():
		total = 30000
		data = np.random.normal(2, 2, total)
		
		can.histogram(data, normalize = True)
		
		can.set_axis(equal_axis = False, legend_on = True)
		can.save(folder + "hist1.jpg", dpi = 200)
		can.clear()
		
		can.ax.hist(data, histtype = "step", align = "mid", density = True)
		
		can.set_axis(equal_axis = False)
		can.save(folder + "hist2.jpg", dpi = 200)
		
		pass
	
	
	main()
	can.close()
	pass
