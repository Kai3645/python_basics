if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	result = (
		cq.Sketch()
		.arc((0, 0), 1.0, 0.0, 360.0)
		.arc((1, 1.5), 0.5, 0.0, 360.0)
		.segment((0.0, 6), (-1, 3.0))
		.hull()  # 凸包(实验体),切线集群
	)
	show_object(result)
