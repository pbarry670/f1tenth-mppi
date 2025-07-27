#!usr/bin/env python3

# A version of the visualization node that works with controlling steering rate instead of steering angle

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
        self.laserscan_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10) #Receive /scan messages - the LiDAR processing is all done in this node to avoid having to publish so many LiDAR points
        self.centerline_sub = self.create_subscription(Float32MultiArray, "/centerline", self.centerline_callback, 10) #Receive the computed centerlien
        self.steercmd_sub = self.create_subscription(Float32MultiArray, "/steercmds", self.steercmd_callback, 10) #Receive the steering rate commands
        self.throttlecmd_sub = self.create_subscription(Float32MultiArray, "/throttlecmds", self.throttlecmd_callback, 10) #Receive the acceleration commands
        self.optimal_traj_index_sub = self.create_subscription(Float32MultiArray, "/optimal_index", self.optimal_index_callback, 10) #Receive the indices of the optimal trajectories
        self.state_sub = self.create_subscription(Float32MultiArray, "/state_vec", self.state_callback, 10) #Receive the published state vector based on previous control inputs

        #Publishers
        self.cost_parameters_pubisher = self.create_publisher(Float32MultiArray, "/cost_params", 10) #Can modify cost parameters in this file and it will change the car's behavior. Included for convenience. When this file runs, the file that carries out CUDA computations takes into account the below parameters for costing.
        self.V_DESIRED = 2.0
        self.VEL_COST_SCALER = 1.0
        self.POS_COST_SCALER = 6.0
        self.HEADING_COST_SCALER = 10.0
        self.CTRL_INPUT_COST_SCALER = 0.0

        #Avoid unused variable name warnings
        self.laserscan_sub
        self.centerline_sub
        self.steercmd_sub
        self.throttlecmd_sub
        self.optimal_traj_index_sub
        self.state_sub

        self.DT = 0.06 #time step when rolling out the dynamics
        self.L_F = 0.1651 #front axle to cg, m
        self.L_R = 0.1651 #rear axle to cg, m

        self.NUM_TRAJECTORIES = 1000 #1000 trajectories of random control inputs
        self.HORIZON_LENGTH = 20 #20 time steps in the time horizon

        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #x, y, heading, longitudinal v, lateral v, steering angle
        self.steerdotcmds = np.tile(1, self.NUM_TRAJECTORIES*self.HORIZON_LENGTH)
        self.throttlecmds = np.tile(1, self.NUM_TRAJECTORIES*self.HORIZON_LENGTH)

        #Initialize things to be plotted to avoid errors
        self.wall = [0.0, 0.0, 0.0, 0.0]
        self.centerline = [0.0, 0.0] 
        self.traj_x = [[0.0, 0.0], [0.0, 0.0]]
        self.traj_y = [[0.0, 0.0], [0.0, 0.0]]



        self.controls_index_list = []
        for i in range(self.NUM_TRAJECTORIES*self.HORIZON_LENGTH): #Form a list of indices from which to index the control inputs for a particular trajectory
            if (i % self.HORIZON_LENGTH == 0):
                self.controls_index_list.append(i)
        self.controls_index_list = np.array(self.controls_index_list)

        self.update_rate = 0.5 #2 Hz
        self.timer = self.create_timer(self.update_rate, self.update_sim) #Update the visualization window every 0.5 seconds

    def update_sim(self):
        self.viz.clear_img() #Wipe the image window
        self.publish_cost_parameters() #Give the cost parameters to the script that calls CUDA
        self.forward_dynamics() #Forward the trajectories in time

        self.viz.draw_car() #The visualization is always centered on the car
        self.viz.draw_pointcloud(self.wall, 1, 255, 0, 0, 1) #pts, radius, R, G, B, thickness. Draws LiDAR readings
        self.viz.draw_polylines(self.traj_x, self.traj_y, self.optimal_index, self.HORIZON_LENGTH, self.NUM_TRAJECTORIES) #Draws trajectories x and y.
        self.viz.draw_centerline(self.centerline, 1, 150, 0, 150, 1) #Draws centerline
        self.viz.draw_pointcloud(self.centerline, 1, 150, 0, 150, 4) #Emphasizes the discrete points on the centerline, which is what the car truly sees.
        
        cv2.imshow("Simulator", self.viz.get_img()) #Produce the image
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
        
        self.wall_left_x = np.zeros(int(num_points/2)) #Initialize variables to store the wall locations
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
        self.wall = np.concatenate((self.wall_x, self.wall_y)) #Format that visualization drawing program takes the wall format in

    def centerline_callback(self, msg : Float32MultiArray): #Process in the centerline
        self.centerline = msg.data

    def steercmd_callback(self, msg : Float32MultiArray): #Process in the steering rate commands
        self.steerdotcmds = msg.data
        self.steerdotcmds = np.array(self.steerdotcmds)

    def throttlecmd_callback(self, msg : Float32MultiArray): #Process in the acceleration commands
        self.throttlecmds = msg.data
        self.throttlecmds = np.array(self.throttlecmds)

    def optimal_index_callback(self, msg : Float32MultiArray): #Process in the indices of the ten lowest-cost trajectories
        self.optimal_index = msg.data
        self.optimal_index = np.array(self.optimal_index)
        
    def state_callback(self, msg : Float32MultiArray): #Process in the current state of the car to act as initial condition for dynamics
        self.state = msg.data
        self.state = np.array(self.state)

    def publish_cost_parameters(self): #Send the cost parameters to the CUDA-calling program to change the car's behavior
        cost_params = [self.V_DESIRED, self.VEL_COST_SCALER, self.POS_COST_SCALER, self.HEADING_COST_SCALER, self.CTRL_INPUT_COST_SCALER]
        self.cost_params_to_publish = Float32MultiArray(data=cost_params)
        self.cost_parameters_pubisher.publish(self.cost_params_to_publish)

    def forward_dynamics(self): #Given an initial state of the car, forward all trajectories in time given their control inputs
        x = np.tile(self.state[0], self.NUM_TRAJECTORIES) #initial conditions for every trajectory
        y = np.tile(self.state[1], self.NUM_TRAJECTORIES)
        psi = np.tile(self.state[2], self.NUM_TRAJECTORIES)
        vx = np.tile(self.state[3], self.NUM_TRAJECTORIES)
        vy = np.tile(self.state[4], self.NUM_TRAJECTORIES)
        steerang = np.tile(self.state[5], self.NUM_TRAJECTORIES)

        self.traj_x = np.empty((self.HORIZON_LENGTH, self.NUM_TRAJECTORIES)) #Prepare variables to store (x, y) of each trajectory
        self.traj_y = np.empty((self.HORIZON_LENGTH, self.NUM_TRAJECTORIES))

        for i in range(self.HORIZON_LENGTH): #Loop through the time horizon
            u_steerdot = np.array(self.steerdotcmds)[self.controls_index_list + i] #index the control inputs for each trajectory
            u_throttle = np.array(self.throttlecmds)[self.controls_index_list + i]

            beta = np.arctan(self.L_R / (self.L_F + self.L_R))*np.tan(steerang)

            x_dot = np.multiply(vx, np.cos(psi + beta))
            y_dot = np.multiply(vx,np.sin(psi + beta))
            psi_dot = np.multiply(np.multiply(vx, np.cos(beta)), np.tan(steerang)) / (self.L_R + self.L_F)

            x = x + x_dot*self.DT #Update the state variables
            y = y + y_dot*self.DT
            psi = psi + psi_dot*self.DT
            vx = vx + u_throttle*self.DT
            steerang = steerang + u_steerdot*self.DT

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
