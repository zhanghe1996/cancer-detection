import os
import json
import urllib2

from utils.config import get

def main():
	im_dir = os.path.join(get("DATA.DATA_PATH"), 'Images')

	with open(os.path.join(get("DATA.DATA_PATH"), 'EyeSnap_2017_v10.json')) as json_file:
		data = json.load(json_file)

	gazes = {'FORWARD_GAZE', 'LEFTWARD_GAZE', 'RIGHTWARD_GAZE', 'UPWARD_GAZE'}

	for case in data['results']:
		if 'LEFT_EYE_DIAGNOSIS' not in case or 'RIGHT_EYE_DIAGNOSIS' not in case:
			continue

		left_dia = case['LEFT_EYE_DIAGNOSIS']
		right_dia = case['RIGHT_EYE_DIAGNOSIS']
		print left_dia, right_dia

		for gaze in gazes:
			if gaze in case:
				im_name = os.path.join(im_dir, case[gaze]['name'] + '.jpeg')
				if not os.path.isfile(im_name):
					url = case[gaze]['url']
					req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
					con = urllib2.urlopen(req)
					with open(im_name, 'wb') as f:
						f.write(con.read())

if __name__ == "__main__":
	main()