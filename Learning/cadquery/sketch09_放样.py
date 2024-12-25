if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	s1 = cq.Sketch().trapezoid(3, 1, 110).vertices().fillet(0.2)
	
	s2 = cq.Sketch().rect(2, 1).vertices().fillet(0.2)
	
	result = cq.Workplane().placeSketch(s1, s2.moved(cq.Location((0, 0, 3)))).loft()  # 放样
	
	show_object(result)
