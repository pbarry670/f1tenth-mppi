#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from dataclasses import dataclass
import cv2
import numpy as np
from .VisualizerO import Visualizer

class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        self.get_logger().info("Visualization node started...")

        self.viz = Visualizer()

        # Subscriptions
        self.laserscan_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.centerline_sub = self.create_subscription(Float32MultiArray, "/centerline", self.centerline_callback, 10)
        self.steercmd_sub = self.create_subscription(Float32MultiArray, "/steercmds", self.steercmd_callback, 10)
        self.throttlecmd_sub = self.create_subscription(Float32MultiArray, "/throttlecmds", self.throttlecmd_callback, 10)

        #Publishers
        self.cost_parameters_pubisher = self.create_publisher(Float32MultiArray, "/cost_params", 10)
        self.V_DESIRED = 2.0
        self.VEL_COST_SCALER = 100.0
        self.POS_COST_SCALER = 10.0
        self.HEADING_COST_SCALER = 10.0
        self.CTRL_INPUT_COST_SCALER = 1.0

        #Avoid unused variable name warnings
        self.laserscan_sub
        self.centerline_sub
        self.steercmd_sub
        self.throttlecmd_sub

        self.DT = 0.02 #time step when rolling out the dynamics
        self.L_F = 0.1651
        self.L_R = 0.1651

        self.state = [0.0, 0.0, 0.0, 1.0, 0.0] #x, y, heading, longitudinal v, lateral v
        self.steercmds = [0.0]
        self.throttlecmds = [0.0]

        #Initialize things to be plotted
        self.wall = [0.0, 0.0, 0.0, 0.0]
        self.centerline = [0.0, 0.0]
        self.trajectory = [[0.0, 0.0], [0.0, 0.0]]

        self.update_rate = 0.05 # 20 Hz
        self.timer = self.create_timer(self.update_rate, self.update_sim)

    def update_sim(self):
        self.viz.clear_img()
        self.publish_cost_parameters()
        self.forward_dynamics()

        self.viz.draw_car()
        self.viz.draw_pointcloud(self.wall, 1, 255, 0, 0, 1) #pts, radius, R, G, B, thickness
        self.viz.draw_pointcloud(self.centerline, 1, 150, 0, 150, 1)
        self.viz.draw_polylines(self.trajectory)

        cv2.imshow("Simulator", self.viz.get_img())
        cv2.waitKey(1)


    def lidar_callback(self, msg : LaserScan):
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

        num_points = int(np.round((self.angle_max - self.angle_min)/self.angle_increment))
        self.wall_left_x = np.zeros(int(num_points/2))
        self.wall_left_y = np.zeros(int(num_points/2))
        self.wall_right_x = np.zeros(int(num_points/2))
        self.wall_right_y = np.zeros(int(num_points/2))

        for i in range(int(num_points/2)):
            angle = self.angle_min + i*self.angle_increment #angle if car is facing straight down the middle of the track
            x_of_left_point = self.lidar_ranges[i]*np.sin(1.5708 + angle) #Relative x position of point on left wall with respect to LiDAR
            y_of_left_point = self.lidar_ranges[i]*np.cos(1.5708 + angle) #Relative y position (ahead of car) of point on left wall with respect to LiDAR

            x_of_right_point = self.lidar_ranges[num_points- 1 - i]*np.cos(-1*angle) #Relative x position of point on right wall with respect to LiDAR
            y_of_right_point = self.lidar_ranges[num_points - 1 - i]*np.sin(angle) #Relative y position of point on right wall with respect to LiDAR

            self.wall_left_x[i] = x_of_left_point
            self.wall_left_y[i] = y_of_left_point
            self.wall_right_x[i] = x_of_right_point
            self.wall_right_y[i] = y_of_right_point 

        self.wall_x = np.concatenate((self.wall_left_x, self.wall_right_x))
        self.wall_y = np.concatenate((self.wall_left_y, self.wall_right_y))
        self.wall = np.concatenate((self.wall_x, self.wall_y))

    def centerline_callback(self, msg : Float32MultiArray):
        self.centerline = msg.data

    def steercmd_callback(self, msg : Float32MultiArray):
        self.steercmds = msg.data

    def throttlecmd_callback(self, msg : Float32MultiArray):
        self.throttlecmds = msg.data

    def publish_cost_parameters(self):
        cost_params = [self.V_DESIRED, self.VEL_COST_SCALER, self.POS_COST_SCALER, self.HEADING_COST_SCALER, self.POS_COST_SCALER]
        self.cost_params_to_publish = Float32MultiArray(data=cost_params)
        self.cost_parameters_pubisher.publish(self.cost_params_to_publish)

    def forward_dynamics(self):
        x = self.state[0]
        y = self.state[1]
        psi = self.state[2]
        vx = self.state[3]
        vy = self.state[4]

        self.traj_x = np.zeros(len(self.steercmds))
        self.traj_y = np.zeros(len(self.throttlecmds))


        for i in range(len(self.steercmds)):
            u_steer = self.steercmds[i]
            u_throttle = self.throttlecmds[i]

            beta = np.arctan(self.L_R / (self.L_F + self.L_R))*np.tan(u_steer)

            x_dot = vx*np.cos(psi + beta)
            y_dot = vx*np.sin(psi + beta)
            psi_dot = (vx*np.cos(beta)*np.tan(u_steer)) / (self.L_R + self.L_F)

            x = x + x_dot*self.DT
            y = y + y_dot*self.DT
            psi = psi + psi_dot*self.DT
            vx = vx + u_throttle*self.DT

            self.traj_x[i] = x
            self.traj_y[i] = y
        
        self.trajectory = np.zeros([len(self.steercmds), 2])
        self.trajectory[:, 0] = self.traj_x
        self.trajectory[:, 1] = self.traj_y

        #print(self.trajectory)



def main(args=None):
    rclpy.init(args=args)
    viz_node = VisualizationNode()

    rclpy.spin(viz_node)

    viz_node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
