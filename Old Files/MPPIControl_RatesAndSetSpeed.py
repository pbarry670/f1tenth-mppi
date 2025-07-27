#!usr/bin/env python3

import numpy as np
from scipy import interpolate
import torch
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_sensor_data

class MPPIControl(Node):

    def __init__(self):

        super().__init__("mppi_rates") #Name of node
        self.mppi = self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos_profile=qos_profile_sensor_data)
            #Inputs are MsgType, /topicName, callback function, and a queue cost setter that allows for packet loss
        ack_qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE, history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, depth=1)
        self.ack_publisher = self.create_publisher(AckermannDriveStamped, "/drive", qos_profile=ack_qos_profile) #Publish AckermannDriveStamped messages to /drive topic
        #self.timer = self.create_timer(0.01, self.publish_ackermann) #Publish ackermann commands every 0.025 seconds

        self.steer_vel = 0.0 #Starting steering angle velocity is zero
        self.accel = 0.0 #Starting acceleration is zero
        self.HORIZON_LENGTH = 30 #Number of time steps over which trajectories are forwarded
        self.NUM_TRAJECTORIES = 8000 #Number of trajectories
        self.DIM_STATE_VEC = 5 #x, y, headAng, lateral V, longitudinal V
        self.DT = 0.02 #Time step when forwarding dynamics over time horizon

        self.L_F = 0.15 #Front axle to cg, m
        self.L_R = 0.15 #Rear axle to cg, m

        self.V_DESIRED = 5 #desired car velocity, m/s
        self.VEL_COST_SCALER = 2.5 #scalar parameter to tune cost associated with deviating from velocity cost

        self.RACELINE_WIDTH = 1 #distance from track boundary to track boundary, m
        self.NUM_RACELINE_PTS = 30 #Number of points on discretized centerline
        self.POS_COST_SCALER = 50 #scalar parameter to tune cost associated with deviation from centerline

        self.HEADING_COST_SCALER = 50 #scalar parameter to tune cost associated with devition from ideal heading angle

        self.CTRL_INPUT_COST_SCALER = 10 #scalar parameter to tune cost associated with magnitude of control inputs
        self.prev_steer_vel_cmd = 0.0
        self.prev_accel_cmd = 0.0

        self.SET_SPEED = 0.5
        self.speed = self.SET_SPEED


        #Algorithm in a Loop:
            #forward_dynamics using x_cuda, u_steering_vel_cuda, u_throttle_cuda
            #While forwarding dynamics, run compute_cost on cost_cuda. Get lidar measurements and use the centerline to compute position and heading cost
            #Get the index of the best trajectory using evaluate_best_cost, this is stored in self.index
            #Obtain the control commands for this trajectory
            #Actuate and reset states, cost
            #Restart the control loop, generating a new warm-started set of random control commands. Send cost back to cuda.
    
    def publish_ackermann(self):
        ack_msg = AckermannDriveStamped() #ROS2 msg type
        ack_msg.header.stamp = rclpy.time.Time().to_msg() #timestamp
        ack_msg.header.frame_id = "laser" #frame ID of message

        ack_msg.drive.steering_angle_velocity = self.steer_vel #steer velocity command, radians/s

        ack_msg.drive.speed = self.speed

        self.ack_publisher.publish(ack_msg) #publish AckermannDriveStamped message to /drive

    def set_steer(self, steer_cmd):
        self.steer = steer_cmd

    def set_accel(self, accel_cmd):
        self.accel = accel_cmd

    def set_steer_vel(self, steer_vel_cmd):
        self.steer_vel = steer_vel_cmd

    def set_speed(self, speed_cmd):
        self.speed = speed_cmd

    def forward_dynamics(self, x_cuda, u_steering_vel_cuda, u_throttle_cuda): #Function to forward the dynamics by one time step; variables already in CUDA should be passed in

        u_steering_cuda = u_steering_vel_cuda*self.DT
        beta = torch.atan((self.L_R/(self.L_F+self.L_R))*torch.tan(u_steering_cuda))

        x_dot = x_cuda[3,:] * torch.cos(x_cuda[2,:] + beta)
        y_dot = x_cuda[3,:] * torch.sin(x_cuda[2,:] + beta)
        headAng_dot = (x_cuda[3,:]*torch.cos(beta)*torch.tan(u_steering_cuda))/(self.L_R + self.L_F)
        x_cuda[0,:] = x_cuda[0,:] + x_dot*self.DT #Update x position of car
        x_cuda[1,:] = x_cuda[1,:] + y_dot*self.DT #Update y position of car
        x_cuda[2,:] = x_cuda[2,:] + headAng_dot*self.DT #Update heading angle of car
        x_cuda[3,:] = self.speed

        return x_cuda

    def state_forward(self, x_vec, u_steer_vel, u_throttle): #Dummy function for testing

        u_steer = u_steer_vel*self.DT
        beta = np.arctan((self.L_R/(self.L_F+self.L_R))*np.tan(u_steer))

        x_dot = x_vec[3]*np.cos(x_vec[2]+ beta)
        y_dot = x_vec[3]*np.sin(x_vec[2]+beta)
        headAng_dot = (x_vec[3]*np.cos(beta)*np.tan(u_steer))/(self.L_R + self.L_F)

        x_new = []
        x_new.append(x_vec[0] + x_dot*self.DT)
        x_new.append(x_vec[1] + y_dot*self.DT)
        x_new.append(x_vec[2] + headAng_dot*self.DT)
        x_new.append(self.speed)
        x_new.append(0) #Lateral velocity

        return x_new

    def compute_cost(self, x_cuda, u_steer_vel, u_throttle):

        pos_cost, heading_cost = self.compute_position_and_heading_cost(x_cuda)
        ctrl_cost = self.CTRL_INPUT_COST_SCALER*(torch.square(u_steer_vel - self.prev_steer_vel_cmd))

        cost = self.VEL_COST_SCALER*torch.square(self.V_DESIRED - x_cuda[3,:]) + self.POS_COST_SCALER*pos_cost + self.HEADING_COST_SCALER*heading_cost + ctrl_cost
        return cost

    #Cost for having command be too far from previous command (both steering and throttle). Big cost for commands out of bounds
        #Or, can just create a cost for the magnitude of the control input. This is what's implemented right now.
    #Cost on deviation from raceline
    #Cost on heading
    #Cost on difference between V_desired and v

    def compute_position_and_heading_cost(self, x_cuda):
        #x_cuda = [x, y, heading, long_v, lat_v)
        x_pos = (x_cuda[0,:]).cpu()
        y_loc = interpolate.splev(x_pos, self.curve)
        y_loc = y_loc.cuda()
        y_dist = y_loc - x_cuda[1,:]
        pos_cost = torch.square((torch.abs(y_dist))/self.RACELINE_WIDTH)

        interpolated_heading = interpolate.splev(x_pos, self.heading_curve)
        interpolated_heading = interpolated_heading.cuda()
        heading_cost = torch.square(x_cuda[2,:] - interpolated_heading)

        return pos_cost, heading_cost

    def evaluate_best_cost(self, cost_cuda): #Takes in cost array and returns index of minimum cost

        self.cost = cost_cuda.cpu()
        self.minCost = min(self.cost)
        for i in range(self.NUM_TRAJECTORIES - 1):
            if self.cost[i] == self.minCost:
                self.index = i
                break
    
    def throttle_to_accel_mapping(self, throttle): #motor model, if needed
    
        accel = throttle 
        return accel
    
    def generate_steering_control_sequence(self, mean, variance): #variance should be such that most trajectories fall within +-(pi/6)

        u_steering = torch.normal(mean, variance, size =(self.HORIZON_LENGTH, self.NUM_TRAJECTORIES) )

        return u_steering #Has HORIZON_LENGTH rows, NUM_TRAJECTORIES columns, Gaussian distribution with configurable mean and variance

    def generate_throttle_control_sequence(self, mean, variance):
    
        u_throttle = torch.normal(mean, variance, size=(self.HORIZON_LENGTH, self.NUM_TRAJECTORIES) )
        return u_throttle #Has HORIZON_LENGTH rows, NUM_TRAJECTORIES columns, Gaussian distribution centered around 0; should range from -1 to 1

    def generate_steering_velocity_control_sequence(self, mean, variance):

        u_steering_vel = torch.normal(mean, variance, size =(self.HORIZON_LENGTH, self.NUM_TRAJECTORIES))
        return u_steering_vel

    def lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

    def compute_centerline(self):
                            
        num_points = int(np.round((self.angle_max - self.angle_min)/self.angle_increment))

        raw_centerline = np.zeros([int(num_points/2), 2]) #Each row is an (x, y) point representing the centerline. x is ahead of car, y is side to side

        for i in range(int(num_points/2)):
            angle = self.angle_min + i*self.angle_increment #angle if car is facing straight down the middle of the track
            x_of_left_point = self.lidar_ranges[i]*np.sin(1.5708 + angle) #Relative x position of point on left wall with respect to LiDAR
            y_of_left_point = self.lidar_ranges[i]*np.cos(1.5708 + angle)*-1 #Relative y position (ahead of car) of point on left wall with respect to LiDAR

            x_of_right_point = self.lidar_ranges[num_points- 1 - i]*np.cos(-1*angle) #Relative x position of point on right wall with respect to LiDAR
            y_of_right_point = self.lidar_ranges[num_points - 1 - i]*np.sin(-1*angle) #Relative y position of point on right wall with respect to LiDAR

            x_of_mid_point = (x_of_right_point + x_of_left_point)/2 #Average of x-coordinates for longitudinal distance away from car
            y_of_mid_point = (y_of_left_point + y_of_right_point)/2 #Average of y-coordinates for lateral distance away from car

            raw_centerline[i,0] = x_of_mid_point
            raw_centerline[i,1] = y_of_mid_point

        #Note: centerline results are less valid at higher heading angles.
        #Note: raw_centerline is nx2, n = (number of LiDAR pts/2). At low n, the centerline points are closer to the car, and at high n, they are further away.
        div_factor = (num_points/2)/self.NUM_RACELINE_PTS
        uncertainty_factor = 6 #Number of NUM_RACELINE_PTS that we choose to not trust (are cut off at the end of the computed centerline). Curvier/narrower track --> more uncertainty
        self.centerline = np.zeros([self.NUM_RACELINE_PTS, 2])
        for i in range(self.NUM_RACELINE_PTS - uncertainty_factor): #30 - 6
            self.centerline[i] = raw_centerline[int(i*div_factor)]

        # THE LINE BELOW IS FOR TESTING ONLY #
        #self.centerline = np.array([[0.0, 0.0], [0.02, 0.0], [0.04, 0.0], [0.06, 0.0], [0.08, 0.0], [0.1, 0.0],[0.12, 0.0], [0.14, 0.0], [0.16, 0.0], [0.18, 0.0], [0.2, 0.0], [0.22, 0.0], [0.24, 0.0], [0.26, 0.0], [0.28, 0.0], [0.3, 0.0], [0.32, 0.0], [0.34, 0.0], [0.37, 0.0], [0.40, 0.0], [0.43, 0.0], [0.46, 0.0], [0.49, 0.0], [0.55, 0.0], [0.68, 0.0], [0.79, 0.0], [0.88, 0.0]]) #straight line
        self.centerline = np.array([[0.0, 0.0], [0.02, 0.0005], [0.04, 0.0021], [0.06, 0.0047], [0.08, 0.0083], [0.1, 0.013],[0.12, 0.0187], [0.14, 0.0255], [0.16, 0.0333], [0.18, 0.0421], [0.2, 0.052], [0.22, 0.0629], [0.24, 0.0749], [0.26, 0.0879], [0.28, 0.1019], [0.3, 0.117], [0.32, 0.1331], [0.34, 0.1503], [0.37, 0.178], [0.40, 0.208], [0.43, 0.2404], [0.46, 0.2751], [0.49, 0.3121], [0.55, 0.3933], [0.68, 0.6011], [0.79, 0.8113], [0.88, 1.007]]) #fits y = 1.3x^2
        # THE LINE ABOVE IS FOR TESTING ONLY #

        centerline_x = self.centerline[:,0]
        centerline_y = self.centerline[:,1]
        last_index = len(centerline_x) - 1

        centerline_x[0:16] = np.sort(centerline_x[0:16])
        for i in range(len(centerline_x) - 1):
            if centerline_x[i+1] < centerline_x[i]:
                last_index = i
                break

        centerline_x = centerline_x[0:last_index]
        centerline_y = centerline_y[0:last_index]

        centerline_dx = centerline_x[1:-1] - centerline_x[0:-2]
        centerline_dy = centerline_y[1:-1] - centerline_y[0:-2]
        centerline_x_short = torch.tensor(centerline_x[0:-2])

        self.curve = interpolate.splrep(centerline_x, centerline_y)
        self.optimal_heading = np.arctan2(centerline_dy, centerline_dx)
        self.heading_curve = interpolate.splrep(centerline_x_short, optimal_heading)

def main(args=None):
    rclpy.init()
    mppi = MPPIControl() #Initialize node mppi
    rclpy.spin_once(mppi) #Press Ctrl + C in terminal to terminate node

    mppi.u_steer_vel = mppi.generate_steering_velocity_control_sequence(0, 1.5) #mean of 0, variance of 1.5 rad/s
    mppi.u_throttle = mppi.generate_throttle_control_sequence(1,1) #mean of 1 m/s^2, variance of 1
    
    mppi.x0 = torch.tensor(np.zeros([mppi.DIM_STATE_VEC, mppi.NUM_TRAJECTORIES])) #Initial state vector array with all trajectories
    
    mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Initialize cost

    start_time = time.time()

    x0_cuda = mppi.x0.cuda()
    u_steer_vel_cuda = mppi.u_steer_vel.cuda()
    u_throttle_cuda = mppi.u_throttle.cuda()
    cost_cuda = mppi.cost.cuda()

    mppi.compute_centerline() #Update centerline before forwarding dynamics

    #Start the run from zero initial conditions
    for i in range(mppi.HORIZON_LENGTH - 1): #Loop through the dynamics with each sequential control input over the horizon
        x_new = mppi.forward_dynamics(x0_cuda, u_steer_vel_cuda[i,:], u_throttle_cuda[i,:])
        cost_cuda = cost_cuda + mppi.compute_cost(x_new, u_steer_vel_cuda[i,:], u_throttle_cuda[i,:]) #Compute cost at each step forward in time
    
    mppi.cost = cost_cuda.cpu()
    index_optimal_trajectory = torch.argmin(mppi.cost) #Select trajectory with the minimal cost
    steer_vel_command = u_steer_vel_cuda[0,index_optimal_trajectory]
    throttle_command = u_throttle_cuda[0, index_optimal_trajectory]

    mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset costs to zero for next projection forward in time

    steer_vel_command = steer_vel_command.cpu()
    throttle_command = throttle_command.cpu()

    steer_vel_command_pub = steer_vel_command.item() #Values to be published to Ackermann
    throttle_command_pub = throttle_command.item()

    mppi.set_steer_vel(steer_vel_command_pub)
    mppi.set_accel(throttle_command_pub)
    mppi.publish_ackermann()

    x_vec = [0,0,0,0,0]
    x_vec = mppi.state_forward(x_vec, steer__vel_command_pub, throttle_command_pub) #Testing dummy for state estimate


    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    x_vec[0] = 0 #Update x location to reflect the fact the origin is centered on car
    flag = True
    timeVec = []

    #while(flag):
    for i in range(100):
        start_time = time.time()
        rclpy.spin_once(mppi)

        x = np.full([mppi.NUM_TRAJECTORIES, mppi.DIM_STATE_VEC], x_vec)
        mppi.x = torch.tensor(np.transpose(x))

        x_cuda = mppi.x.cuda()
        mppi.u_steer_vel = mppi.generate_steering_velocity_control_sequence(steer_vel_command_pub, 1.5)
        mppi.u_throttle = mppi.generate_throttle_control_sequence(throttle_command_pub, 1)

        mppi.compute_centerline() #generate nx2 array -- (x, y) -- of raceline points here (the center of the track) from LIDAR

        u_steer_vel_cuda = mppi.u_steer_vel.cuda()
        u_throttle_cuda = mppi.u_throttle.cuda()
        cost_cuda = mppi.cost.cuda()

        for i in range(mppi.HORIZON_LENGTH - 1):
            x_new = mppi.forward_dynamics(x_cuda, u_steer_vel_cuda[i,:], u_throttle_cuda[i,:])
            cost_cuda = cost_cuda + mppi.compute_cost(x_new, u_steer_vel_cuda[i,:], u_throttle_cuda[i,:])

        mppi.cost = cost_cuda.cpu()
        index_optimal_trajectory = torch.argmin(mppi.cost)
        steer_vel_command = u_steer_vel_cuda[0, index_optimal_trajectory].cpu()
        throttle_command = u_throttle_cuda[0, index_optimal_trajectory].cpu()

        steer_vel_command_pub = steer_vel_command.item() #Values to be published to Ackermann
        throttle_command_pub = throttle_command.item()

        mppi.set_steer_vel(steer_vel_command_pub)
        mppi.set_accel(throttle_command_pub)
        mppi.publish_ackermann()

        mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset Cost
        x_vec = mppi.state_forward(x_vec, steer_vel_command_pub, throttle_command_pub) #State estimation for testing
        x_vec[0] = 0 #This and the next line reflect the fact that the LIDAR always comes from the origin
        print(x_vec)
        #print(f"Throttle Command: {throttle_command_pub}")
        #print(f"Steering Velocity Command: {steer_vel_command_pub}")
        end_time = time.time()
        timeVec.append(end_time - start_time)

    print(sum(timeVec)/len(timeVec))



if __name__ == '__main__':
    main()