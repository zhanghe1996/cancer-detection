import re
import os
import cv2
import csv
import json
import argparse
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt
import xml.dom.minidom as minidom

from utils.config import get

m = get('DIAGNOSIS_MAP')
eye_side = get("TRAIN.EYE_SIDE")
pupil_side = get("TRAIN.PUPIL_SIDE")
# feature_dim = 3 * pupil_side ** 2 + 1
feature_dim = 3 * eye_side ** 2 + 3 * pupil_side ** 2 + 1 + 2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--test', dest='test', action='store_true')

    args = parser.parse_args()

    return args

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

def order_compare(x, y):
	if x[feature_dim - 3] == 0 or y[feature_dim - 3] == 0:
		return 1

	if x[feature_dim - 3] > y[feature_dim - 3]:
		return 1
	elif x[feature_dim - 3] == y[feature_dim - 3]:
		if x[feature_dim - 2] > y[feature_dim - 2]:
			return 1
		else:
			return -1
	else:
		return -1

def main():
	args = parse_args()

	# get config
	if args.test:
		im_dir = os.path.join(get("DATA.DATA_PATH"), 'test_data', 'Images')
		ann_dir = os.path.join(get("DATA.DATA_PATH"), 'test_data', 'Annotations')
		image_sets = os.path.join(get("DATA.DATA_PATH"), 'test_data', 'ImageSets', 'train.txt')
	else:
		im_dir = os.path.join(get("DATA.DATA_PATH"), 'Images')
		ann_dir = os.path.join(get("DATA.DATA_PATH"), 'Annotations')
		image_sets = os.path.join(get("DATA.DATA_PATH"), 'ImageSets', 'train.txt')

	# initialize diagnosis map
	diagnosis_map = {}
	object_id_map = {}

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

			object_id_map[case[gaze]['name']] = case['objectId']

	# get all image names from train.txt
	with open(image_sets) as f:
		image_names = f.read().splitlines()

	output = [[0] * feature_dim]
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
		
		# assert(num_objs == 2)
		if num_objs != 2:
			continue

		pupil_boxes = np.zeros((num_objs, 4), dtype = np.uint16)
		labels = np.zeros(num_objs, dtype = np.uint16)

		left_right_strings = np.array(['left', 'right'])

		for ix, obj in enumerate(objs):
			labels[ix] = diagnosis_map[image_name][ix]
			pupil_boxes[ix][0] = int(get_data_from_tag(obj, 'xmin'))
			pupil_boxes[ix][1] = int(get_data_from_tag(obj, 'ymin'))
			pupil_boxes[ix][2] = int(get_data_from_tag(obj, 'xmax'))
			pupil_boxes[ix][3] = int(get_data_from_tag(obj, 'ymax'))
		
		# exchange boxes when pupil_boxes[0] is actually the right pupil
		if pupil_boxes[0][0] > pupil_boxes[1][0]:
			pupil_boxes[[0, 1]] = pupil_boxes[[1, 0]]
			left_right_strings[[0, 1]] = left_right_strings[[1, 0]]

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

			feature = feature.tolist()
			feature.append(object_id_map[image_name])
			feature.append(left_right_strings[i])
			feature.append(labels[i])

			assert(len(feature) == feature_dim)

			output.append(feature)

	if args.test:
		output = sorted(output, cmp=order_compare)
	else:
		shuffle(output)
	
	if args.test: 
		with open(get("DATA.TEST_CSV_PATH"), 'wb') as csv_file:
		    wr = csv.writer(csv_file, lineterminator='\n')
		    wr.writerows(output)
	else:
		with open(get("DATA.CSV_PATH"), 'wb') as csv_file:
		    wr = csv.writer(csv_file, lineterminator='\n')
		    wr.writerows(output)

if __name__ == "__main__":
	main()