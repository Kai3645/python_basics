if __name__ == '__main__':
	import os
	import sys
	import subprocess

	folder = sys.argv[1]
	mov_name = sys.argv[2]
	fps_set = 15 if len(sys.argv) < 4 else int(sys.argv[3])

	input_path = folder + os.sep + mov_name
	output_path = folder + os.sep + mov_name.replace(".", "_") + ".gif"

	palette_gen = [
		'ffmpeg', '-i', input_path,
		'-vf', f'fps={fps_set},scale={720}:-1:flags=lanczos,palettegen',
		'-y', 'palette.png'
	]

	gif_gen = [
		'ffmpeg', '-i', input_path, '-i', 'palette.png',
		'-lavfi', f'fps={fps_set},scale={720}:-1:flags=lanczos [x]; [x][1:v] paletteuse',
		'-loop', '0', '-y', output_path
	]

	subprocess.run(palette_gen, check = True)
	subprocess.run(gif_gen, check = True)
