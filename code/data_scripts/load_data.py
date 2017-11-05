import re
import os
import cv2
import numpy as np
from PIL import Image
import xml.dom.minidom as minidom

from utils.config import get


def get_data_from_tag(node,tag):
	return node.getElementsByTagName(tag)[0].childNodes[0].data

def main():
	im_dir = os.path.join(get("DATA.DATA_PATH"), 'Images')
	xml_dir = os.path.join(get("DATA.DATA_PATH"), 'Annotations')
	image_sets = os.path.join(get("DATA.DATA_PATH"), 'ImageSets', 'train.txt')

	image_side = get("TRAIN.IMAGE_SIDE")
	pupil_side = get("TRAIN.PUPIL_SIDE")

	with open(image_sets) as f:
		image_names = f.read().splitlines()

	output = None
	for image_name in image_names:
		im_file = os.path.join(im_dir, image_name + '.jpeg')
		xml_file = os.path.join(xml_dir, image_name + '.xml')

		im = Image.open(im_file)
		downsampled_im = im.resize((image_side, image_side), Image.ANTIALIAS)
		im_pix = np.array(downsampled_im, dtype = np.float32)
		im_pix = np.reshape(im_pix, (-1))

		# im = cv2.imread(im_file)
		# downsampled_im = cv2.resize(im, (image_side, image_side))
		# im_pix = np.array(downsampled_im, dtype = np.float32)
		# im_pix = np.reshape(im_pix, (-1))

		with open(xml_file) as f:
			data = minidom.parseString(f.read())

		objs = data.getElementsByTagName('object')
		num_objs = len(objs)
		assert(num_objs == 2)
		boxes = np.zeros((num_objs, 4), dtype = np.uint16)
		labels = np.zeros(num_objs, dtype = np.uint16)

		for ix, obj in enumerate(objs):
			if get_data_from_tag(obj, 'name') == "cancer":
				labels[ix] = 1

			boxes[ix][0] = int(get_data_from_tag(obj, 'xmin'))
			boxes[ix][1] = int(get_data_from_tag(obj, 'ymin'))
			boxes[ix][2] = int(get_data_from_tag(obj, 'xmax'))
			boxes[ix][3] = int(get_data_from_tag(obj, 'ymax'))
			
		if boxes[0][0] > boxes[1][0]:
			boxes[[0, 1]] = boxes[[1, 0]]
			labels[[0, 1]] = labels[[1, 0]]

		for box in boxes:
			pupil = im.crop(box);
			pupil = pupil.resize((pupil_side, pupil_side), Image.ANTIALIAS)
			
			# pupil = im[box[1] : box[3], box[0] : box[2]]
			# pupil = cv2.resize(pupil, (pupil_side, pupil_side))

			pupil_pix = np.array(pupil, dtype = np.float32)
			pupil_pix = np.reshape(pupil_pix, (-1))
			im_pix = np.append(im_pix, pupil_pix)

		im_pix = im_pix / 127.5 - 1
		im_pix = np.append(im_pix, labels)
		assert(im_pix.shape[0] == 3 * image_side ** 2 + 6 * pupil_side ** 2 + 2)
		
		if output is None:
			output = im_pix
		else:
			output = np.vstack((output, im_pix))

	np.random.shuffle(output)
	output = np.vstack((np.zeros(3 * image_side ** 2 + 6 * pupil_side ** 2 + 2), output))
	np.savetxt(get("DATA.CSV_PATH"), output, delimiter = ",", fmt = "%f")

if __name__ == "__main__":
	main()