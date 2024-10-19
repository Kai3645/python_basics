import re

if __name__ == '__main__':
	path_in = "/Users/mini/Downloads/save10.hg"
	path_out = "/Users/mini/Downloads/save10_out.txt"


	def main():
		fr = open(path_in, "rb")
		fw = open(path_out, "w")

		data = str(fr.read())
		print(data)

		# rtn = re.split(r"([a-zA-Z0-9.,:/|]+])", data)
		# for s in rtn[:3]:
		# 	print(s)


	main()
	pass
