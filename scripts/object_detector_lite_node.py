#!/usr/bin/env python3

import rospy
import sys
import math
from math import floor
import message_filters
import cv2
from object_detection.msg import BoundingBox, BoundingBoxes, Detection, Detections, Lidar_Point, Lidar_Points
import sensor_msgs
from cv_bridge import CvBridge
import numpy as np
import time
from collections import deque

import gc

## Poser imports
from keras import models, layers, Sequential
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageOps, ImageColor, ImageDraw, ImageFont
import glob
import matplotlib.pyplot as plt
import json
import cv2
import torch

## OCR imports
import pytesseract
import re

os.system('export CUDA_VISIBLE_DEVICES=""')

class Object_Detector_Lite():
	def __init__(self):
		self.cur_xres = -1
		self.cur_yres = -1
		
		self.roi_x = 0.5
		self.roi_y = 0.5
		self.roi_size = 0.9
		
		self.stopsign_ocr = True
		
		#self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
		self.model = torch.hub.load('/home/actor/yolov5', 'yolov5n', source='local')
		
		self.image_topic = "/camera/image_raw"
		self.boxes_topic = "/object_detector/detects_raw"
		
		self.image_sub = rospy.Subscriber(self.image_topic, sensor_msgs.msg.Image, self.image_callback)
		self.boxes_pub = rospy.Publisher(self.boxes_topic, BoundingBoxes, queue_size = 10)
		self.debug_image_pub = rospy.Publisher('/object_detector/debug_image', sensor_msgs.msg.Image, queue_size = 10)
		
		self.rate = rospy.Rate(30)
		
		self.bridge = CvBridge()
		
		gc.enable()
		print('Object Detector Lite is running')
		rospy.spin()
		#while not rospy.is_shutdown():
		#	self.rate.sleep()
	
	def read_ocr(self, inImage):
		height = inImage.shape[0]
		width = inImage.shape[1]
		ocr_image = inImage[round(0.25*height):round(0.75*height),round(0.05*width):round(0.95*width)]
		ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
		ocr_image = cv2.medianBlur(ocr_image, 5)
		ocr_image = cv2.threshold(ocr_image, 150, 255, cv2.THRESH_BINARY)
		ocr_image = ocr_image[1]
		
		self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(ocr_image, 'passthrough'))
		
		return pytesseract.image_to_string(ocr_image, config='--psm 6')
	
	def image_callback(self, inImage):
		im_width = inImage.width
		im_height = inImage.height
		
		roi_x0 = max(0, round(self.roi_x*im_width - self.roi_size*im_height/2))
		roi_x1 = min(im_width, round(roi_x0 + self.roi_size*im_height))
		roi_y0 = max(0, round(self.roi_y*im_height - self.roi_size*im_height/2))
		roi_y1 = min(im_height, round(roi_y0 + self.roi_size*im_height))
		roi_xw = roi_x1 - roi_x0
		roi_yw = roi_y1 - roi_y0
		
		frame = self.bridge.imgmsg_to_cv2(inImage, 'bgr8')
		
		detections = self.model(frame)
		result = detections.pred[0]
		num_detections = result.shape[0]
		labels = detections.names
		
		boxes_msg = BoundingBoxes()
		boxes_msg.header = inImage.header
		boxes_msg.image_header = inImage.header
		tmp_max = 0
		tmp_maxbox = None
		for i in range(num_detections):
			tmp_box = BoundingBox()
			tmp_box.probability = result[i, 4].item()
			tmp_box.xmin = round(result[i, 0].item())
			tmp_box.ymin = round(result[i, 1].item())
			tmp_box.xmax = round(result[i, 2].item())
			tmp_box.ymax = round(result[i, 3].item())
			tmp_box.id = result[i, 5].int().item()
			tmp_box.label = labels[result[i, 5].int()]
			if tmp_box.label == "stop sign" and self.stopsign_ocr:
				ocr_cropped = frame[tmp_box.ymin:tmp_box.ymax, tmp_box.xmin:tmp_box.xmax]
				ocr_result = self.read_ocr(ocr_cropped)
				ocr_result = re.sub(r'\W+', '', ocr_result)
				print(f'OCR Result: {ocr_result}')
				tmp_box.label = f'{tmp_box.label}-{ocr_result}'
				print(f'Label: {tmp_box.label}')
			#print(type(tmp_box.probability))
			#print(type(tmp_box.xmin))
			#print(type(tmp_box.ymin))
			#print(type(tmp_box.xmax))
			#print(type(tmp_box.ymax))
			#print(type(tmp_box.id))
			#print(type(tmp_box.Class))
			#print()
			#if tmp_box.Class == "Person" and tmp_box.probability > tmp_max:
			if tmp_box.probability > 0.0:# and tmp_box.Class == "Person":
				boxes_msg.bounding_boxes.append(tmp_box)
				#tmp_max = tmp_box.probability
				#tmp_maxbox = tmp_box
		
		#if not tmp_maxbox == None:
		#	boxes_msg.bounding_boxes.append(tmp_maxbox)
		#print('Callback Heartbeat')
		#test_image = cv2.cvtColor(tf.cast(image, dtype=tf.uint8).numpy(), 
		self.boxes_pub.publish(boxes_msg)
		

if __name__ == '__main__':
	rospy.init_node('object_detector_lite_node')
	print('Object Detector Lite is initialized')
	
	try:
		Object_Detector_Lite()
	except rospy.ROSInterruptException:
		pass

