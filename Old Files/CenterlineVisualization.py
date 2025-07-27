#!usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from geometry_msgs.msg import Vector3, Point, Pose
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray


class LidarTesting(Node):

    def __init__(self):
        super().__init__("lidar_to_centerline") #Name of node
        self.lidar_to_centerline = self.create_subscription(LaserScan,"/scan", self.lidar_callback, qos_profile=qos_profile_sensor_data)
        self.lidar_to_centerline = self.create_publisher(MarkerArray, "/computed_centerline", qos_profile=qos_profile_sensor_data)

    def lidar_callback(self, msg:LaserScan):
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

        num_points = int(np.round((self.angle_max - self.angle_min)/self.angle_increment))

        raw_centerline = np.zeros([360, 2]) #Each row is an (x, y) point representing the centerline. x is ahead of car, y is side to side

        for i in range(360):
            angle = self.angle_min + i*self.angle_increment #angle if car is facing straight down the middle of the track
            y_of_left_point = self.lidar_ranges[i]*np.cos(angle)*-1 #Relative y position of point on left wall with respect to LiDAR
            x_of_left_point = self.lidar_ranges[i]*np.sin(angle)*-1 #Relative x position (ahead of car) of point on left wall with respect to LiDAR

            y_of_right_point = self.lidar_ranges[720 - 1 - i]*np.cos(angle) #Relative y position of point on right wall with respect to LiDAR
            x_of_right_point = self.lidar_ranges[720 - 1 - i]*np.sin(angle) #Relative x position of point on right wall with respect to LiDAR

            x_of_mid_point = (x_of_right_point + x_of_left_point)/2 #Average of x-coordinates for longitudinal distance away from car
            y_of_mid_point = (y_of_left_point + y_of_right_point)/2 #Average of y-coordinates for lateral distance away from car

            raw_centerline[i,0] = x_of_mid_point
            raw_centerline[i,1] = y_of_mid_point

        #Note: centerline results are less valid at higher heading angles.
        #Note: raw_centerline is nx2, n = (number of LiDAR pts/2). At low n, the centerline points are closer to the car, and at high n, they are further away.

        div_factor = (360)//30 #Factor by which discretized centerline will be shorter than raw centerline. Must be divisible by (num_points/2)
        self.centerline = np.zeros([(360)//div_factor, 2])
        for i in range(360//div_factor):
            self.centerline[i] = raw_centerline[i*div_factor]

        self.send_centerline()

    def send_centerline(self):
        self.markerArray = MarkerArray()
        colorsGroup = []
        triplePoints = []   
        
        for i in range(30):
            point = Point()
            point.x = self.centerline[i,0]
            point.y = self.centerline[i,1]
            point.z = 0.1
            marker = Marker()
            marker.header.frame_id = "/laser"
            marker.type = marker.POINTS
            marker.action = marker.ADD
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = 1.0
            marker.pose.orientation.w = 1.0
            triplePoints.append(point)
            marker.points = triplePoints
            self.markerArray.markers.append(marker)
          

        self.lidar_to_centerline.publish(self.markerArray)

    
def main(args=None):
    rclpy.init()
    node = LidarTesting()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
