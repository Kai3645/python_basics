if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	"""
	mode type:
	    a: 融合
	    s: 切割
	    i: 相交
	    r: 替换
	    c: 仅存储以进行构造,需要tag,类似草图
	"""
	result = (
		cq.Sketch()
		.rect(1, 2, mode = "c", tag = "base")
		.vertices(tag = "base")
		.circle(0.7)
		.reset()
		.edges("|Y", tag = "base")
		.ellipse(1.2, 1, mode = "i")
		.reset()
		.rect(2, 2, mode = "i")
		.clean()  # 移除杂边
	)
	show_object(result)
