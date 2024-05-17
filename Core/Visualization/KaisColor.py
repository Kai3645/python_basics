import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from Core.Basic import NUM_ERROR


class KaisColor:
	"""
	my favorite cnames
	crimson, orangered, darkorange, gold, yellowgreen,
	forestgreen, turquoise, deepskyblue, dodgerblue, royalblue,
	slateblue, darkorchid, purple, orchid, violet, pink
	"""
	axis_cnames = ("crimson", "forestgreen", "royalblue")

	@classmethod
	def axis(cls, idx: int):
		return cls.plotColor(cls.axis_cnames[idx])

	@staticmethod
	def rand(start: int = 35, end: int = 250, *, length: int = 1):
		colors = np.random.random((length, 3))
		valid = np.sum(colors, axis = 1) < 1
		while np.all(valid):
			length = len(np.where(valid)[0])
			colors[valid, :] = np.random.random((length, 3))
			valid = np.sum(colors, axis = 1) < 1
		colors = colors * (end - start) + start
		if length > 1: return colors
		return colors[0]

	@classmethod
	def plotColor(cls, name, gbr: bool = False, lib_type = "CSS"):
		color_format = {
			"CSS": (mcolors.CSS4_COLORS, ""),
			"TAB": (mcolors.TABLEAU_COLORS, "tab:"),
			"XKCD": (mcolors.XKCD_COLORS, "xkcd:"),
		}
		color_lib, affix = color_format[lib_type]

		names = np.atleast_1d(name)
		c_hex = [color_lib[affix + cn] for cn in names]
		colors = np.atleast_2d(cls.hex2rgb(c_hex))

		if gbr: colors = colors[:, ::-1]
		if len(names) > 1: return colors
		return colors[0]

	@staticmethod
	def plotColorMap(name, length: int, is_gbr: bool = False):
		cmap = plt.get_cmap(name, length)
		colors = np.zeros((length, 3))
		for i in range(length):
			colors[i, :] = cmap(i)[:3]
		colors *= 255
		if is_gbr: return colors[:, ::-1]
		return colors

	@staticmethod
	def invert(color):
		return 255 - color

	@staticmethod
	def hex2rgb(c_hex):
		c_hex = np.atleast_1d(c_hex)
		length = len(c_hex)
		c_rgb = np.zeros((length, 3))
		c_v = np.asarray([int(a[1:], 16) for a in c_hex])
		c_rgb[:, 0] = np.right_shift(c_v, 16)
		c_v = np.bitwise_and(c_v, 0xffff)
		c_rgb[:, 1] = np.right_shift(c_v, 8)
		c_rgb[:, 2] = np.bitwise_and(c_v, 0xff)

		if length > 1: return c_rgb
		return c_rgb[0]

	@staticmethod
	def rgb2hex(c_rgb):
		c_rgb = np.atleast_2d(c_rgb).astype(int)
		length = len(c_rgb)

		c_v = np.left_shift(c_rgb[:, 0], 16)
		c_v = np.bitwise_or(c_v, np.left_shift(c_rgb[:, 1], 8))
		c_v = np.bitwise_or(c_v, c_rgb[:, 2])
		c_hex = [f"#{a:06x}" for a in c_v]

		if length > 1: return c_hex
		return c_hex[0]

	@classmethod
	def hist_color(cls, x, grad: int = 50, name: str = "viridis", is_gbr: bool = False):
		"""
		:param x: (N, ) array
		:param name:
		:param grad: gradient
		:param is_gbr:
		:return: colors: (N, 3) array
		"""
		idxes = np.asarray(x) - min(x)
		idxes /= (max(idxes) + NUM_ERROR) / grad
		idxes = idxes.astype(int)
		base_colors = cls.plotColorMap(name, grad, is_gbr)
		return base_colors[idxes]


def draw_color_tables(folder: str, darkmode = False):
	if folder[-1] != "/": folder += "/"
	cell_width = 200
	cell_height = 24
	swatch_width = 50
	margin = 18
	top_margin = 43
	dpi = 72

	if darkmode: plt.style.use('dark_background')

	def color_table(colors, title, del_affix = ""):
		data = np.asarray(list(colors.items()))
		c_hsv = mcolors.rgb_to_hsv(KaisColor.hex2rgb(data[:, 1]) / 256)
		tmp_data = sorted(((tuple(c), tuple(n)) for (c, n) in zip(c_hsv, data)))
		data = np.asarray(tmp_data, object)[:, 1]

		n = len(data)
		tmp_row = int(math.sqrt(n * 10))
		clm = int(round(n / tmp_row))
		row = int(round(n / clm))
		print(f"{title}, counts = {n}, shape = {(row, clm)}")

		w = cell_width * clm + 2 * margin
		h = cell_height * row + margin + top_margin

		fig, ax = plt.subplots(figsize = (w / dpi, h / dpi), dpi = dpi)
		fig.subplots_adjust(margin / w, margin / h, (w - margin) / w, (h - top_margin) / h)
		ax.set_xlim(0, cell_width * clm)
		ax.set_ylim(cell_height * (row - 0.5), -cell_height / 2.)
		ax.yaxis.set_visible(False)
		ax.xaxis.set_visible(False)
		ax.set_axis_off()
		ax.set_title(title, fontsize = 22, loc = "left", pad = 10)

		for i, cd in enumerate(data):
			x = i // row
			y = i % row * cell_height

			x0 = cell_width * x
			x1 = cell_width * x + swatch_width
			x2 = cell_width * x + swatch_width + 7

			ax.hlines(y, x0, x1, linewidth = 18,
			          color = cd[1])

			ax.text(x2, y, cd[0].replace(del_affix, ""), fontsize = 13,
			        horizontalalignment = 'left',
			        verticalalignment = 'center')

		return fig

	f = color_table(mcolors.TABLEAU_COLORS, "Tableau Palette", del_affix = "tab:")
	f.savefig(folder + "Tableau_Palette.pdf")

	f = color_table(mcolors.CSS4_COLORS, "CSS Colors")
	f.savefig(folder + "CSS_Colors.pdf")

	f = color_table(mcolors.XKCD_COLORS, "XKCD Colors", del_affix = "xkcd:")
	f.savefig(folder + "XKCD_Colors.pdf")


if __name__ == '__main__':
	def main():
		folder = "/home/kai/PycharmProjects/pyCenter/d_2022_0728/out"
		draw_color_tables(folder, True)
		pass


	main()
	pass
