import re
import os
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom

from utils.config import get

m = get('DIAGNOSIS_MAP')

def get_data_from_tag(node,tag):
	return node.getElementsByTagName(tag)[0].childNodes[0].data

def getDiagnosisIndex(diagnosis):
	if diagnosis in m:
		return m[diagnosis]
	else:
		return len(m) - 1

def getEyeIndex(eye_boxes, pupil_box):
	for i, eye_box in enumerate(eye_boxes):
		if eye_box[0] <= pupil_box[0] and eye_box[1] <= pupil_box[1] and eye_box[2] >= pupil_box[2] and eye_box[3] >= pupil_box[3]:
			return i
	return -1

def main():
	# get config
	im_dir = os.path.join(get("DATA.DATA_PATH"), 'Images')
	ann_dir = os.path.join(get("DATA.DATA_PATH"), 'Annotations')
	image_sets = os.path.join(get("DATA.DATA_PATH"), 'ImageSets', 'train.txt')

	eye_side = get("TRAIN.EYE_SIDE")
	pupil_side = get("TRAIN.PUPIL_SIDE")
	feature_dim = 3 * eye_side ** 2 + 3 * pupil_side ** 2 + 1
	# feature_dim = 3 * pupil_side ** 2 + 1
	
	# initialize diagnosis map
	diagnosis_map = {}

	with open(os.path.join(get("DATA.DATA_PATH"), 'EyeSnap_2017_v10.json')) as json_file:
		data = json.load(json_file)

	gazes = {'FORWARD_GAZE', 'LEFTWARD_GAZE', 'RIGHTWARD_GAZE', 'UPWARD_GAZE'}
	for case in data['results']:
		if 'LEFT_EYE_DIAGNOSIS' not in case or 'RIGHT_EYE_DIAGNOSIS' not in case:
			continue

		for gaze in gazes:
			if gaze not in case:
				continue
			diagnosis_map[case[gaze]['name']] = [
				getDiagnosisIndex(case['RIGHT_EYE_DIAGNOSIS']), 
				getDiagnosisIndex(case['LEFT_EYE_DIAGNOSIS'])
			]

	# get all image names from train.txt
	with open(image_sets) as f:
		image_names = f.read().splitlines()

	output = None
	for image_name in image_names:
		if image_name not in diagnosis_map:
			continue

		print image_name
		im_file = os.path.join(im_dir, image_name + '.jpeg')
		pupil_file = os.path.join(ann_dir, 'pupil', image_name + '.xml')
		eye_file = os.path.join(ann_dir, 'eye', image_name + '_eye.npy')

		if not os.path.isfile(pupil_file):
			continue

		im = Image.open(im_file)

		with open(pupil_file) as f:
			data = minidom.parseString(f.read())

		objs = data.getElementsByTagName('object')
		num_objs = len(objs)
		assert(num_objs == 2)
		pupil_boxes = np.zeros((num_objs, 4), dtype = np.uint16)
		labels = np.zeros(num_objs, dtype = np.uint16)

		for ix, obj in enumerate(objs):
			labels[ix] = diagnosis_map[image_name][ix]
			pupil_boxes[ix][0] = int(get_data_from_tag(obj, 'xmin'))
			pupil_boxes[ix][1] = int(get_data_from_tag(obj, 'ymin'))
			pupil_boxes[ix][2] = int(get_data_from_tag(obj, 'xmax'))
			pupil_boxes[ix][3] = int(get_data_from_tag(obj, 'ymax'))
		
		# exchange boxes when pupil_boxes[0] is actually the right pupil
		if pupil_boxes[0][0] > pupil_boxes[1][0]:
			pupil_boxes[[0, 1]] = pupil_boxes[[1, 0]]

		eye_boxes = np.load(eye_file)

		for i in range(2):
			pupil = im.crop(pupil_boxes[i])
			pupil = pupil.resize((pupil_side, pupil_side), Image.ANTIALIAS)
			pupil_pix = np.array(pupil, dtype = np.float32)

			# pupil_pix = 255 - pupil_pix
			# plt.imshow(pupil_pix)
			# plt.show()

			pupil_pix = np.reshape(pupil_pix, (-1))
			pupil_pix = pupil_pix / 127.5 - 1

			eye_index = getEyeIndex(eye_boxes, pupil_boxes[i])

			if eye_index == -1:
				print "Cannot find corresponding eye box"
				print pupil_boxes[i]
				print np.array(eye_boxes, dtype=np.int32)
				continue

			eye = im.crop(eye_boxes[eye_index][:-1])
			eye = eye.resize((eye_side, eye_side), Image.ANTIALIAS)
			eye_pix = np.array(eye, dtype = np.float32)

			eye_pix = np.reshape(eye_pix, (-1))
			eye_pix = eye_pix / 127.5 - 1

			feature = np.append(eye_pix, pupil_pix)
			feature = np.append(feature, labels[i])
			assert(feature.shape[0] == feature_dim)

			if output is None:
				output = feature
			else:
				output = np.vstack((output, feature))

	np.random.shuffle(output)
	output = np.vstack((np.zeros(feature_dim), output))
	np.savetxt(get("DATA.CSV_PATH"), output, delimiter = ",", fmt = "%f")

if __name__ == "__main__":
	main()