#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32
from dataclasses import dataclass
import cv2
import numpy as np
from .Visualizer import Visualizer

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
        self.optimal_traj_index_sub = self.create_subscription(Float32MultiArray, "/optimal_index", self.optimal_index_callback, 10)
        self.state_sub = self.create_subscription(Float32MultiArray, "/state_vec", self.state_callback, 10)

        #Publishers
        self.cost_parameters_pubisher = self.create_publisher(Float32MultiArray, "/cost_params", 10)
        self.V_DESIRED = 2.0
        self.VEL_COST_SCALER = 1.0
        self.POS_COST_SCALER = 1.0
        self.HEADING_COST_SCALER = 10.0
        self.CTRL_INPUT_COST_SCALER = 0.1

        #Avoid unused variable name warnings
        self.laserscan_sub
        self.centerline_sub
        self.steercmd_sub
        self.throttlecmd_sub
        self.optimal_traj_index_sub
        self.state_sub

        self.DT = 0.02 #time step when rolling out the dynamics
        self.L_F = 0.1651
        self.L_R = 0.1651

        self.NUM_TRAJECTORIES = 1000
        self.HORIZON_LENGTH = 20

        self.state = [0.0, 0.0, 0.0, 0.0, 0.0] #x, y, heading, longitudinal v, lateral v
        self.steercmds = np.tile(1, self.NUM_TRAJECTORIES*self.HORIZON_LENGTH)
        self.throttlecmds = np.tile(1, self.NUM_TRAJECTORIES*self.HORIZON_LENGTH)

        #Initialize things to be plotted
        self.wall = [0.0, 0.0, 0.0, 0.0]
        self.centerline = [0.0, 0.0]
        self.traj_x = [[0.0, 0.0], [0.0, 0.0]]
        self.traj_y = [[0.0, 0.0], [0.0, 0.0]]



        self.controls_index_list = []
        for i in range(self.NUM_TRAJECTORIES*self.HORIZON_LENGTH):
            if (i % self.HORIZON_LENGTH == 0):
                self.controls_index_list.append(i)
        self.controls_index_list = np.array(self.controls_index_list)
        print(self.controls_index_list)

        #self.update_rate = 0.05 # 20 Hz
        self.update_rate = 0.5 #2 Hz
        self.timer = self.create_timer(self.update_rate, self.update_sim)

    def update_sim(self):
        self.viz.clear_img()
        self.publish_cost_parameters()
        self.forward_dynamics()

        self.viz.draw_car()
        self.viz.draw_pointcloud(self.wall, 1, 255, 0, 0, 1) #pts, radius, R, G, B, thickness
        self.viz.draw_polylines(self.traj_x, self.traj_y, self.optimal_index, self.HORIZON_LENGTH, self.NUM_TRAJECTORIES)
        self.viz.draw_centerline(self.centerline, 1, 150, 0, 150, 1)
        self.viz.draw_pointcloud(self.centerline, 1, 150, 0, 150, 4)
        
        cv2.imshow("Simulator", self.viz.get_img())
        cv2.waitKey(1)


    def lidar_callback(self, msg : LaserScan):
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

        self.lidar_ranges = np.array(self.lidar_ranges)
        self.lidar_ranges = self.lidar_ranges[:-180] #Remove first and last 180 points to make lidar "see" only from -pi/2 to pi/2 angles
        self.lidar_ranges = self.lidar_ranges[180:]
        self.angle_min = -1.57079632679
        self.angle_max = 1.57079632679
        
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
        self.steercmds = np.array(self.steercmds)

    def throttlecmd_callback(self, msg : Float32MultiArray):
        self.throttlecmds = msg.data
        self.throttlecmds = np.array(self.throttlecmds)

    def optimal_index_callback(self, msg : Float32MultiArray):
        self.optimal_index = msg.data
        self.optimal_index = np.array(self.optimal_index)
        
    def state_callback(self, msg : Float32MultiArray):
        self.state = msg.data
        self.state = np.array(self.state)

    def publish_cost_parameters(self):
        cost_params = [self.V_DESIRED, self.VEL_COST_SCALER, self.POS_COST_SCALER, self.HEADING_COST_SCALER, self.CTRL_INPUT_COST_SCALER]
        self.cost_params_to_publish = Float32MultiArray(data=cost_params)
        self.cost_parameters_pubisher.publish(self.cost_params_to_publish)

    def forward_dynamics(self):
        x = np.tile(self.state[0], self.NUM_TRAJECTORIES)
        y = np.tile(self.state[1], self.NUM_TRAJECTORIES)
        psi = np.tile(self.state[2], self.NUM_TRAJECTORIES)
        vx = np.tile(self.state[3], self.NUM_TRAJECTORIES)
        vy = np.tile(self.state[4], self.NUM_TRAJECTORIES)

        self.traj_x = np.empty((self.HORIZON_LENGTH, self.NUM_TRAJECTORIES))
        self.traj_y = np.empty((self.HORIZON_LENGTH, self.NUM_TRAJECTORIES))

        for i in range(self.HORIZON_LENGTH):
            u_steer = np.array(self.steercmds)[self.controls_index_list + i]
            u_throttle = np.array(self.throttlecmds)[self.controls_index_list + i]

            beta = np.arctan(self.L_R / (self.L_F + self.L_R))*np.tan(u_steer)

            x_dot = np.multiply(vx, np.cos(psi + beta))
            y_dot = np.multiply(vx,np.sin(psi + beta))
            psi_dot = np.multiply(np.multiply(vx, np.cos(beta)), np.tan(u_steer)) / (self.L_R + self.L_F)

            x = x + x_dot*self.DT
            y = y + y_dot*self.DT
            psi = psi + psi_dot*self.DT
            vx = vx + u_throttle*self.DT

            self.traj_x[i, :] = x
            self.traj_y[i, :] = y

        
        



def main(args=None):
    rclpy.init(args=args)
    viz_node = VisualizationNode()

    rclpy.spin(viz_node)

    viz_node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
