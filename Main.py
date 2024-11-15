#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from Image import *
from Utils import *

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from math import *
from collections import deque

font = cv2.FONT_HERSHEY_SIMPLEX
direction = 0
Images=[]
N_SLICES = 4
paused = False  # 일시 정지 여부

for q in range(N_SLICES):
    Images.append(Image())


class lane_detect():
    def __init__(self):
        # self.bridge = CvBridge()
        # rospy.init_node('lane_detection_node', anonymous=False)
        # rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        # self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cap = cv2.VideoCapture("/home/kwcccc/polygon/omo1.mp4")
        if not self.cap.isOpened():
            print("Error: 비디오 파일을 열 수 없습니다.")
            exit()
        else:
            self.lane_detect()

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.camera_processing()
    
    def camera_processing(self):
        frame = self.image
        cv2.namedWindow("VideoFrame")
        cv2.moveWindow('VideoFrame', 700, 0)
        cv2.imshow("VideoFrame", frame)

        warpped_img, minv = warpping(frame)
        cv2.namedWindow('BEV')
        cv2.moveWindow('BEV', 0, 0)
        cv2.imshow('BEV', warpped_img)  
        # print(frame.shape) # # (h, w, c) = (480, 640, 3)
        blurred_img = cv2.GaussianBlur(warpped_img, (7, 7), 5)
        cv2.namedWindow('Blurred')
        cv2.moveWindow('Blurred', 350, 0)
        cv2.imshow('Blurred', blurred_img)
        # array = np.frombuffer(data, dtype='uint8')
        # img = cv2.imdecode(array, 1)


        img = blurred_img[:, :]
        direction = 0
        img = RemoveBackground(img, False)
        line_distance = np.zeros((4, ), dtype=int)
        line_contourX = np.zeros((4, ), dtype=int)
        line_contourY = np.linspace(0, img.shape[0], 4)
        if img is not None:
            SlicePart(img, Images, N_SLICES)
            for i in range(N_SLICES):
                # direction += Images[i].dir
                line_distance[i] = Images[i].dir
                line_contourX[i] = Images[i].contourCenterX
            fm = RepackImages(Images)
            cv2.imshow("Vision Race", fm)
        
        # 'Space' 키 입력을 기다리며 일시 정지/재생
        key = cv2.waitKey(33) & 0xFF   

        k = 0.001
        amp = 2

        line_points = np.stack((line_contourX, line_contourY), axis=1)
        print(line_distance)
        line_angle = None
        if line_points[1, 0] == line_points[2, 0]:
            line_angle = 0.0
        else:
            slope = (line_points[1, 1] - line_points[2, 1]) / (line_points[1, 0] - line_points[2, 0])
            line_angle = degrees(atan(slope))

        distance = np.mean(line_distance)

        theta_err = radians(line_angle)
        lat_err = distance
        
        speed = Twist()
        speed.linear.x = 0.1
        speed.angular.z = theta_err + atan(k*lat_err)
        self.pub.publish(speed)
        # print(degrees(theta_err),degrees(atan(k*lat_err)))
        # print(speed.angular.z)# Clean up the connection

    def video_processing(self):
        while True:

            ret, frame = self.cap.read()
            cv2.namedWindow("VideoFrame")
            cv2.moveWindow('VideoFrame', 700, 0)
            cv2.imshow("VideoFrame", frame)

            warpped_img, minv = warpping(frame)
            cv2.namedWindow('BEV')
            cv2.moveWindow('BEV', 0, 0)
            cv2.imshow('BEV', warpped_img)  
            # print(frame.shape) # # (h, w, c) = (480, 640, 3)
            blurred_img = cv2.GaussianBlur(warpped_img, (7, 7), 5)
            cv2.namedWindow('Blurred')
            cv2.moveWindow('Blurred', 350, 0)
            cv2.imshow('Blurred', blurred_img)
            # array = np.frombuffer(data, dtype='uint8')
            # img = cv2.imdecode(array, 1)


            img = blurred_img[:, :]
            direction = 0
            img = RemoveBackground(img, False)
            line_distance = np.zeros((4, ), dtype=int)
            line_contourX = np.zeros((4, ), dtype=int)
            line_contourY = np.linspace(0, img.shape[0], 4)
            if img is not None:
                SlicePart(img, Images, N_SLICES)
                for i in range(N_SLICES):
                    # direction += Images[i].dir
                    line_distance[i] = Images[i].dir
                    line_contourX[i] = Images[i].contourCenterX
                fm = RepackImages(Images)
                cv2.imshow("Vision Race", fm)
            if not ret:
                print("End of video")
                break
            
            # 'Space' 키 입력을 기다리며 일시 정지/재생
            key = cv2.waitKey(33) & 0xFF

            if key == ord(' '):  # 스페이스바가 눌리면 일시 정지/재생 토글
                paused = not paused
            
            # 영상이 일시 정지 상태라면
            while paused:
                # 스페이스바가 눌려야만 영상 재생
                key = cv2.waitKey(33) & 0xFF
                if key == ord(' '):  # 스페이스바를 다시 눌러 재생
                    paused = False
            
            # 'q'를 눌러서 영상 종료
            if key == ord('q'):
                print("Video paused. Press 'q' to exit.")
                break
            

            k = 0.001
            amp = 2

            line_points = np.stack((line_contourX, line_contourY), axis=1)
            print(line_distance)
            line_angle = None
            if line_points[1, 0] == line_points[2, 0]:
                line_angle = 0.0
            else:
                slope = (line_points[1, 1] - line_points[2, 1]) / (line_points[1, 0] - line_points[2, 0])
                line_angle = degrees(atan(slope))

            distance = np.mean(line_distance)

            theta_err = radians(line_angle)
            lat_err = distance
            
            # speed = Twist()
            # speed.linear.x = 0.1
            # speed.angular.z = theta_err + atan(k*lat_err)
            # self.pub.publish(speed)
            # print(degrees(theta_err),degrees(atan(k*lat_err)))
            # print(speed.angular.z)# Clean up the connection
        self.cap.release()
        cv2.destroyAllWindows()
