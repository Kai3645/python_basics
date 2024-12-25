if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	s = cq.Sketch().trapezoid(3, 1, 110).vertices().fillet(0.2)
	
	result = (
		cq.Workplane()
		.box(5, 5, 5)
		.faces(">X")
		.workplane()
		.transformed((0, 0, -90))  # 变换
		.placeSketch(s)
		.cutThruAll()  # 切割通孔
	)
	show_object(result)
