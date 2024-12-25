if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	result = (
		cq.Workplane()
		.box(5, 5, 1)  # 盒子
		.faces(">Z")  # 选择面
		.sketch()  # 草图
		.regularPolygon(2, 3, tag = "outer")  # 正多边形
		.regularPolygon(1.5, 3, mode = "s")
		.vertices(tag = "outer")
		.fillet(0.2)  # 圆角
		.finalize()  # 结束草图
		.extrude(0.5)  # 拉伸
	)
	show_object(result)
