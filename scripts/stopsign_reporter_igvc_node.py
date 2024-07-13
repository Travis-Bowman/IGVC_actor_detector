#!/usr/bin/env python3

import rospy
import sys
import math
import std_msgs
from math import floor
from object_detection.msg import BoundingBox, BoundingBoxes
import numpy as np
import time

import gc

import os

class Stopsign_Reporter_IGVC():
	def __init__(self):
		self.target_label = "stop sign-STOP"
		self.boxes_topic = "/object_detector/detects_raw"
		
		self.decay = 0.95
		self.detects_sub = rospy.Subscriber(self.boxes_topic, BoundingBoxes, self.detection_callback)
		self.stop_pub = rospy.Publisher("/object_detector/sign_is_stop", std_msgs.msg.Bool, queue_size = 10)
		self.is_stop = 0
		
		self.rate = rospy.Rate(10)
		print('IGVC STOP Reporter is running')
		#rospy.spin()
		while not rospy.is_shutdown():
			tmp_msg = std_msgs.msg.Bool()
			tmp_msg.data = self.is_stop > 0.1
			self.stop_pub.publish(tmp_msg)
			self.rate.sleep()
	
	def detection_callback(self, boxes):
		for box in boxes.bounding_boxes:
			if box.label == "stop sign-STOP":
				self.is_stop = 1
				return
		self.is_stop = self.decay*self.is_stop
		

if __name__ == '__main__':
	rospy.init_node('stopsign_reporter_igvc_node')
	print('IGVC STOP Reporter is initialized')
	
	try:
		Stopsign_Reporter_IGVC()
	except rospy.ROSInterruptException:
		pass

