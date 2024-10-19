import numpy as np

if __name__ == '__main__':
	import os
	import re

	folder = os.path.dirname(__file__)

	path_in = "/Users/mini/Downloads/无人深空/save10.hg"
	path_out = "/Users/mini/Downloads/无人深空/save10_out.txt"


	def split_string(data):
		pattern = rb"([\{\}\[\]:,])"
		lines = []

		lines_d1 = re.split(rb"(\".*?\":)", data)
		# for line_d1 in lines_d1:
		# 	if line_d1 == b"": continue
		# 	if line_d1[0] == b"\"":
		# 		lines.append(line_d1)
		# 		continue
		# 	lines_d2 = re.split(pattern, line_d1)
		# 	for line_d2 in lines_d2:
		# 		line_d2 = line_d2.strip()
		# 		if line_d2 == b"": continue
		# 		lines.append(line_d2)
		return lines_d1
		pass


	def get_list(lines, ip0):
		if lines[ip0] != b"[": return None, -1

		data = []

		ip1 = ip0 + 1
		while ip1 < len(lines):
			if lines[ip1] == b"]":
				ip1 += 1
				break
			if lines[ip1] == b",":
				ip1 += 1
				continue
			if lines[ip1] == b"{":
				d, ip1 = get_dict(lines, ip1)
				data.append(d)
				continue
			if lines[ip1] == b"[":
				d, ip1 = get_list(lines, ip1)
				data.append(d)
				continue
			data.append(lines[ip1])
			ip1 += 1
		if len(data) > 1 and type(data[-1]) is str:
			data.sort()
		return data, ip1


	def get_dict(lines, ip0):
		if lines[ip0] != b"{": return None, -1

		name = lines[ip0 + 1]
		data = {name: None}

		ip1 = ip0 + 3
		while ip1 < len(lines):
			if ip1 < 0: return None, -1
			if lines[ip1] == b"}":
				# print(name, data[name])
				return data, ip1 + 1
			if lines[ip1] == b",":
				name = lines[ip1 + 1]
				data[name] = None
				ip1 += 3
				continue
			if lines[ip1] == b"{":
				data[name], ip1 = get_dict(lines, ip1)
				# print(name, data[name])
				continue
			if lines[ip1] == b"[":
				data[name], ip1 = get_list(lines, ip1)
				# print(name, data[name])
				continue

			data[name] = lines[ip1]
			# print(name, data[name])
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
				out_str += head + str(d) + ",\n"
		out_str += "  " * dp + "]\n"
		return out_str


	def str_dict(data, dp = 0):
		head = "  " * dp + "  "
		out_str = "{\n"
		k_list = list(data.keys())
		k_list.sort()
		for k in k_list:
			out_str += head + str(k) + ": "
			if type(data[k]) is dict:
				out_str += str_dict(data[k], dp + 1)
			elif type(data[k]) is list:
				out_str += str_list(data[k], dp + 1)
			else:
				out_str += str(data[k]) + ",\n"
		out_str += "  " * dp + "}\n"
		return out_str


	def main():
		fr = open(path_in, "rb")
		fw = open(path_out, "w")

		data = fr.read()

		print(data[:200])

		lines = split_string(data)

		for line in lines:
			fw.write(str(line)[2:-1] + "\n")

		fr.close()
		fw.close()


	main()
