if __name__ == '__main__':
	import difflib


	def main():
		path_01 = "/Users/mini/Downloads/无人深空远征/accountdata_14_out.txt"
		path_02 = "/Users/mini/Downloads/无人深空远征/accountdata_15_out.txt"
		path_diff = "/Users/mini/Downloads/无人深空远征/accountdata_1415.txt"

		lines_01 = open(path_01, "r").readlines()
		lines_02 = open(path_02, "r").readlines()
		rtn = difflib.ndiff(lines_01, lines_02)

		with open(path_diff, "w") as fw:
			fw.writelines(rtn)


	main()
	pass
