if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	result = (
		cq.Sketch()
		.segment((0, 0), (-4, 8.0), "s1")
		.arc((0.0, 3.0), (1.5, 1.5), (0.0, 0.0), "a1")
		.constrain("s1", "Fixed", None)  # 指定点-固定
		.constrain("s1", "a1", "Coincident", None)  # 指定点-重合
		.constrain("a1", "s1", "Coincident", None)
		.constrain("s1", "a1", "Angle", 150)  # 指定切线角度
		.solve()
		.segment((0, 0), (0, 4.0))
		.segment((4.0, 0))
		.close()
		.segment((1.5, 1.5), (-1, 2.0))
		.close()
		.assemble()
	)
	show_object(result)
