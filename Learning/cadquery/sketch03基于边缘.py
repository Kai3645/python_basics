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
		.segment((0.0, 0), (0.0, 2.0))
		.segment((2.0, 0))
		.close()  # 闭合线段
		.arc((0.6, 0.6), 0.5, 0.0, 360.0)  # 圆弧
		.assemble(tag = "face")
		.edges("%LINE", tag = "face")
		.vertices()
		.chamfer(0.2)  # 倒角
	)
	show_object(result)
