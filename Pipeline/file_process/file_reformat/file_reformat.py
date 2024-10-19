if __name__ == '__main__':
	import os
	import sys

	path_in = "/Users/mini/Downloads/无人深空远征1-13/accountdata_org.hg"
	path_out = "/Users/mini/Downloads/无人深空远征1-13/accountdata_org2.hg"

	folder = os.path.dirname(__file__)
	config_path = folder + "/symbol_config.ymal"

	import yaml
	import re

	print(">> loading .. ")
	if len(sys.argv) == 4:
		temp_path = sys.argv[3]
		if os.path.exists(temp_path):
			config_path = temp_path

	print(config_path)
	config = yaml.load(open(config_path), Loader = yaml.SafeLoader)

	stair_count = 0

	stair_up_key = []
	stair_down_key = []
	for a, b in config["Indentation_Key"]:
		stair_up_key.append(a)
		stair_down_key.append(b)

	new_line_key = [] + config["NewLine_Key"]
	interval_key = [] + config["Interval_Key"]

	print(stair_up_key)
	print(stair_down_key)
	print(new_line_key)
	print(interval_key)

	total_keys = stair_up_key + stair_down_key + new_line_key + interval_key
	total_keys = list(set(total_keys))

	print(total_keys)
	symbols = r"(["
	for i, k in enumerate(total_keys):
		if k in ["*", ".", "?", "+" "^", "$", "|", "\\", "/", "[", "]", "(", ")", "{", "}"]:
			symbols += '\\' + k
		else: symbols += k
	symbols += r"])"
	print(symbols)

	print("------")
	with open(path_in, "r") as fr:
		fw = open(path_out, "w")
		lines = []
		for line in fr.readlines():
			lines_dp1 = re.split(r"(\"[^\"]*\")", line)
			for line_dp1 in lines_dp1:
				if line_dp1[0] == "\"":
					lines.append(line_dp1)
					continue
				lines_dp2 = re.split(r"(\'[^\']*\')", line_dp1)
				for line_dp2 in lines_dp2:
					if line_dp2[0] == "\'":
						lines.append(line_dp2)
						continue
					line_dp2 = line_dp2.strip("\n\r")
					line_dp2 = line_dp2.replace("\t", "")
					lines_dp3 = re.split(symbols, line_dp2)
					for line_dp3 in lines_dp3:
						line_dp3 = line_dp3.strip()
						if line_dp3 == "": continue
						lines.append(line_dp3)

		print(lines[:20])
		for i, line in enumerate(lines):
			if line[0] == "\"" or line[0] == "\'" or line in interval_key:
				fw.write(" ")
				fw.write(line)
				continue
			elif line in new_line_key:
				fw.write(line)
				if i < len(lines) - 1 and lines[i + 1] in stair_up_key:
					continue
				fw.write("\n")
				fw.write("    " * stair_count)
			elif line in stair_up_key:
				stair_count += 1
				fw.write(line)
				if i < len(lines) - 1 and lines[i + 1] in stair_up_key:
					continue
				fw.write("\n")
				fw.write("    " * stair_count)
			elif line in stair_down_key:
				stair_count -= 1
				fw.write("\n")
				fw.write("    " * stair_count)
				fw.write(line)
