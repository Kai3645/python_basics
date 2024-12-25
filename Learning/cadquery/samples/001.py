if __name__ == "__main__":
	"""
	选定面,对外缘边进行offset
	"""
	
	import cadquery as cq
	from cadquery.vis import show_object
	
	box = (
		cq.Workplane()
		.box(5, 4, 3)
		.edges("|Z").fillet(1)
	)
	show_object(box)
	
	edge = (
		box
		.faces(">Z")
		.wires()
		.toPending()
		.extrude(2, False)
		.faces("+Z or -Z").shell(-0.4)
	)
	show_object(edge)
	
	tmp = (
		cq.Assembly()
		.add(box, name = "box", color = cq.Color("red"))
		.add(edge, name = "edge", color = cq.Color("green"))
	)
	tmp.constrain("box@faces@>Y", "edge@faces@>X", "Plane")
	tmp.solve()
	show_object(tmp)
