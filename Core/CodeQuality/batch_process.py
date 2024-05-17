# def video2image_batch(video_path:str, ):
#
#
# 	folder_input, folder_output, name_format: str, video_type: str
# 	if folder_input[-1] != "/": folder_input += "/"
# 	if folder_output[-1] != "/": folder_output += "/"
#
# 	names = listdir(folder_input, pattern = "*." + video_type)
# 	Times = np.empty(0)
# 	Paths = np.empty(0)
#
# 	if name_format.split(".")[-1] != video_type:
# 		name_format += "." + video_type
#
# 	time_ = 0
# 	for n in names:
# 		dt = datetime.strptime(n, name_format)
# 		dt = dt.replace(tzinfo = timezone(timedelta(hours = SYSTEM_TIME_ZONE)))
# 		dt = dt.astimezone(tz = timezone.utc)
# 		start_time = weekTimestamp(dt)
#
# 		time_flag = False
# 		if start_time < time_:
# 			start_time = time_
# 			time_flag = True
#
# 		local_folder = n.split(".")[0]
# 		t_list, p_list = video2image(
# 			video_path = folder_input + n,
# 			folder_out = mkdir(folder_output, local_folder),
# 			start_time = start_time,
# 		)
# 		if time_flag:
# 			t_list = t_list[1:]
# 			p_list = p_list[1:]
# 		p_list = local_folder + "/" + p_list
# 		Times = np.concatenate((Times, t_list))
# 		Paths = np.concatenate((Paths, p_list))
#
# 		time_ = Times[-1]
#
# 	write_timeList(folder_output + "time_list.txt", Times, Paths)
# 	return Times, Paths
