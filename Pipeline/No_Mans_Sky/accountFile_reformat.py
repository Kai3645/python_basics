import numpy as np

if __name__ == '__main__':
	import os
	import re

	folder = os.path.dirname(__file__)

	path_in = "/Users/mini/Downloads/无人深空远征/accountdata_15.hg"
	path_out = "/Users/mini/Downloads/无人深空远征/accountdata_15_out.txt"


	def split_string(lines_raw):
		pattern = r"([\{\}\[\]:,])"
		lines = []
		for line_d0 in lines_raw:
			line_d0 = line_d0.strip("\n\r")
			line_d0 = line_d0.replace("\t", "")
			lines_d1 = re.split(r"(\"[^\"]*\")", line_d0)
			for line_d1 in lines_d1:
				if line_d1[0] == "\"":
					lines.append(line_d1)
					continue
				lines_d2 = re.split(pattern, line_d1)
				for line_d2 in lines_d2:
					line_d2 = line_d2.strip()
					if line_d2 == "": continue
					lines.append(line_d2)
		return lines
		pass


	def get_list(lines, ip0):
		if lines[ip0] != "[": return None, -1

		data = []

		ip1 = ip0 + 1
		while ip1 < len(lines):
			if lines[ip1] == "]":
				ip1 += 1
				break
			if lines[ip1] == ",":
				ip1 += 1
				continue
			if lines[ip1] == "{":
				d, ip1 = get_dict(lines, ip1)
				data.append(d)
				continue
			if lines[ip1] == "[":
				d, ip1 = get_list(lines, ip1)
				data.append(d)
				continue
			data.append(lines[ip1])
			ip1 += 1
		if len(data) > 1 and type(data[-1]) is str:
			data.sort()
		return data, ip1


	def get_dict(lines, ip0):
		if lines[ip0] != "{": return None, -1

		name = lines[ip0 + 1]
		data = {name: None}

		ip1 = ip0 + 3
		while ip1 < len(lines):
			if lines[ip1] == "}":
				return data, ip1 + 1
			if lines[ip1] == ",":
				name = lines[ip1 + 1]
				data[name] = None
				ip1 += 3
				continue
			if lines[ip1] == "{":
				data[name], ip1 = get_dict(lines, ip1)
				continue
			if lines[ip1] == "[":
				data[name], ip1 = get_list(lines, ip1)
				continue
			data[name] = lines[ip1]
			ip1 += 1
		return data, ip1


	def str_list(data, dp = 0):
		head = "  " * dp + "  "
		out_str = "[\n"
		for d in data:
			if type(d) is dict:
				out_str += head + str_dict(d, dp + 1)
			elif type(d) is list:
				out_str += head + str_list(d, dp + 1)
			else:
				out_str += head + d + ",\n"
		out_str += "  " * dp + "]\n"
		return out_str


	def str_dict(data, dp = 0):
		head = "  " * dp + "  "
		out_str = "{\n"
		k_list = list(data.keys())
		k_list.sort()
		for k in k_list:
			out_str += head + k + ": "
			if type(data[k]) is dict:
				out_str += str_dict(data[k], dp + 1)
			elif type(data[k]) is list:
				out_str += str_list(data[k], dp + 1)
			else:
				out_str += data[k] + ",\n"
		out_str += "  " * dp + "}\n"
		return out_str


	def main():
		fr = open(path_in, "r")
		fw = open(path_out, "w")

		lines = split_string(fr.readlines())

		data, _ = get_dict(lines, 0)

		out_str = str_dict(data)
		fw.write(out_str)

		fr.close()
		fw.close()


	main()
