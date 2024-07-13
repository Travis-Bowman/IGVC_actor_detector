#!/usr/bin/env python3

import rospy
import sys
import math
import std_msgs
from math import floor
from object_detection.msg import Detection, Detections, Lidar_Point, Lidar_Points
import numpy as np
import time

import gc

import os

class Detection_Reporter_IGVC():
	def __init__(self):
		self.target_label = "person"
		self.boxes_topic = "/object_detector/detects_nolidar"
		
		self.detects_sub = rospy.Subscriber(self.boxes_topic, Detections, self.detection_callback)
		self.target_sub = rospy.Subscriber("/object_detector/target_label", std_msgs.msg.String, self.target_callback)
		self.dist_pub = rospy.Publisher("/object_detector/closest_dist", std_msgs.msg.Float64, queue_size = 10)
		self.detections = None
		
		self.rate = rospy.Rate(10)
		print('IGVC Detection Reporter is running')
		#rospy.spin()
		while not rospy.is_shutdown():
			if not self.detections == None:
				self.detection_act(self.detections)
			self.rate.sleep()
	
	def target_callback(self, newTarget):
		self.target_label = newTarget.data
	
	def detection_callback(self, detections):
		self.detections = detections
	
	def detection_act(self, detections):
		tmp_min_dist = -1
		for det in detections.detections:
			if det.obj_class == self.target_label:
				if (tmp_min_dist < 0) or (det.lidar_point.distance < tmp_min_dist):
					tmp_min_dist = det.lidar_point.distance
		dist_msg = std_msgs.msg.Float64()
		dist_msg.data = tmp_min_dist
		self.dist_pub.publish(dist_msg)
		

if __name__ == '__main__':
	rospy.init_node('detection_reporter_igvc_node')
	print('IGVC Detection Reporter is initialized')
	
	try:
		Detection_Reporter_IGVC()
	except rospy.ROSInterruptException:
		pass

