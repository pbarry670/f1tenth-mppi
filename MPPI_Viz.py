#!usr/bin/env python3
import numpy as np
import torch
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
import time_horizon_cuda

class MPPIControl(Node):

    def __init__(self):

        super().__init__("mppiviz") #Name of node
        self.mppi_viz = self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos_profile=qos_profile_sensor_data)
            #Inputs are MsgType, /topicName, callback function, and a queue cost setter that allows for packet loss
        ack_qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE, history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, depth=1)
        self.ack_publisher = self.create_publisher(AckermannDriveStamped, "/ackermann_cmd", qos_profile=ack_qos_profile) #Publish AckermannDriveStamped messages to /ackermann_cmd topic
        self.centerline_publisher = self.create_publisher(Float32MultiArray, "/centerline", 10) #Publish the computed centerline to /centerline topic
        self.steering_publisher = self.create_publisher(Float32MultiArray, "/steercmds", 10) #Publish the steering rate commands to /steercmds
        self.throttle_publisher = self.create_publisher(Float32MultiArray, "/throttlecmds", 10) #Publish the acceleration commands to /throttlecmds
        self.optimal_traj_index_publisher = self.create_publisher(Float32MultiArray, "/optimal_index", 10) #Publish the indices of the lowest-cost trajectories to /optimal_index
        self.state_publisher = self.create_publisher(Float32MultiArray, "/state_vec", 10) #Publish the state of the car to /state_vec. Because the car always acts as the origin, the only nonzero values are steering rate and vx.

        self.cost_params_sub = self.create_subscription(Float32MultiArray, "/cost_params", self.cost_params_callback, 10) #Subscribe to /cost_params, a topic that sets V_DESIRED, and cost scalers for position, heading, velocity, and control inputs costs.
        self.cost_params_sub #Avoid unused variable warning

        self.steer = 0.0 #Starting steering angle is zero
        self.accel = 0.0 #Starting acceleration is zero
        self.speed = 0.0 #Starting speed is zero
        self.steer = 0.0 # Starting steer angle is zero
        self.prev_steer_cmd = 0.0 #At startup, all previous inputs are zero
        self.prev_accel_cmd = 0.0
        self.prev_speed = 0.0
        self.prev_steer = 0.0

        self.max_steer = 0.3 #Maximum magnitude steering angle the car may actuate to
        self.max_speed = 2.0 #Maximum speed the car may reach, included for safety
        self.min_speed = 0.2 #Minimum speed the car is allowed to travel, included to prevent car stalling at speeds near zero

        self.HORIZON_LENGTH = 20 #Number of time steps over which trajectories are forwarded
        self.NUM_TRAJECTORIES = 1000 #Number of trajectories
        self.DIM_STATE_VEC = 6 #x, y, headAng, longitudinal V, lateral V, steering angle
        self.trueDT = 0.0 #Time step used for first actuation; updated to be true time step after first iteration

        self.L_F = 0.1651 #Front axle to cg, m
        self.L_R = 0.1651 #Rear axle to cg, m
        self.NUM_RACELINE_PTS = 60 # Number of points in the centerline

        self.list_of_indices = [] #Needed for processing 2D control input arrays into 1D arrays
        for i in range(self.HORIZON_LENGTH):
            self.list_of_indices.append(i)
        self.list_of_indices = np.array(self.list_of_indices)

        self.num_cost_indices = 10 #Used for finding the ten lowest-cost trajectories
        self.opimal_index = np.tile(1, self.num_cost_indices)

        ## Initialize cost parameters, though these can be changed when Visualization_Node.py publishes to /cost_params
        self.V_DESIRED = 2.0 #desired car velocity, m/s
        self.VEL_COST_SCALER = 1.0 #scalar parameter to tune cost associated with deviating from velocity cost
        self.POS_COST_SCALER = 6.0 #scalar parameter to tune cost associated with deviation from centerline
        self.HEADING_COST_SCALER = 10.0 #scalar parameter to tune cost associated with devition from ideal heading angle
        self.CTRL_INPUT_COST_SCALER = 1.0 #scalar parameter to tune cost associated with magnitude of control inputs
        
    
    def publish_ackermann(self):
        ack_msg = AckermannDriveStamped() #ROS2 msg type
        ack_msg.header.stamp = rclpy.time.Time().to_msg() #timestamp
        ack_msg.header.frame_id = "laser" #frame ID of message
        ack_msg.drive.steering_angle= -1*self.steer #steer command, radians. Uses -1 because servo sign conventions are opposite the LiDAR's.
        ack_msg.drive.speed = self.speed #speed command, m/s^2
        self.ack_publisher.publish(ack_msg) #publish AckermannDriveStamped message to /ackermann_cmd

    def publish_centerline(self):
        self.centerline_to_publish = Float32MultiArray(data=self.centerline_to_publish)
        self.centerline_publisher.publish(self.centerline_to_publish)

    def publish_steercmds(self):
        #self.steercmds_to_publish is a self.HORIZON_LENGTH x self.NUM_TRAJECTORIES array
        #Reformat the 2D array self.steercmds_to_publish into a 1D array, self.intermediate_steercmds:
        self.intermediate_steercmds = np.empty((self.HORIZON_LENGTH*self.NUM_TRAJECTORIES))
        for i in range(self.NUM_TRAJECTORIES):
            self.intermediate_steercmds[i*self.HORIZON_LENGTH + self.list_of_indices] = self.steercmds_to_publish[:, i]

        self.steercmds_to_publish = Float32MultiArray(data=self.intermediate_steercmds) # Place the steering rate commands in a Float32MultiArray ROS2 msg type
        self.steering_publisher.publish(self.steercmds_to_publish) #Publish the steering rate commands to /steercmds

    def publish_throttlecmds(self):
        #self.throttlecmds_to_publish is a self.HORIZON_LENGTH x self.NUM_TRAJECTORIES array
        #Reformat the 2D array self.throttlecmds_to_publish into a 1D array, self.intermediate_steercmds:
        self.intermediate_throttlecmds = np.empty((self.HORIZON_LENGTH*self.NUM_TRAJECTORIES))
        for i in range(self.NUM_TRAJECTORIES):
            self.intermediate_throttlecmds[i*self.HORIZON_LENGTH + self.list_of_indices] = self.throttlecmds_to_publish[:,i]

        self.throttlecmds_to_publish = Float32MultiArray(data=self.intermediate_throttlecmds) #Place the accel commands in a Float32MultiArray ROS2 msg type
        self.throttle_publisher.publish(self.throttlecmds_to_publish) #Publish the accel commands to /throttlecmds

    def publish_optimal_traj_index(self):
        self.index_to_publish = Float32MultiArray(data=self.optimal_index) #Place the indices of the lowest-cost trajectories in a Float32MultiArray ROS2 msg type
        self.optimal_traj_index_publisher.publish(self.index_to_publish) #Publish the lowest-cost indices to /optimal_index
        
    def publish_state_vec(self):
        self.state = [0.0, 0.0, 0.0, self.speed, 0.0, self.steer] #[x, y, psi, vx, vy, steerang]
        self.state_to_publish = Float32MultiArray(data=self.state) #Place the state vector in a Float32MultiArray ROS2 msg
        self.state_publisher.publish(self.state_to_publish) #Publish the state vector to /state_vec

    def cost_params_callback(self, msg : Float32MultiArray):
        self.cost_params = msg.data # Grab the cost paramaters from the message published to /cost_params
        
        ## self.cost_params = [V_DESIRED, VEL_COST_SCALER, POS_COST_SCALER, _HEADING_COST_SCALER, CTRL_INPUT_COST_SCALER]
        self.V_DESIRED = self.cost_params[0] 
        self.VEL_COST_SCALER = self.cost_params[1]
        self.POS_COST_SCALER = self.cost_params[2]
        self.HEADING_COST_SCALER = self.cost_params[3]
        self.CTRL_INPUT_COST_SCALER = self.cost_params[4]

    def set_steer(self, steerdot_cmd): #Update steering angle of car using time since last actuation and steering rate command
        steer_increment = steerdot_cmd*self.trueDT #Find change in steering angle from current steering angle
        self.steer = self.prev_steer + steer_increment #Update steering angle with increment
          
        if (self.steer >= self.max_steer): #Regulate steering angle to within maximum bounds
            self.steer = self.max_steer
        if (self.steer <= -self.max_steer):
            self.steer = -self.max_steer

        self.prev_steer = self.steer #Previous steer now holds value of current steer for the next iteration to happen

    def set_speed(self, accel_cmd): #Update the speed of the car using time since last actuation and acceleration command
        self.accel = accel_cmd
        speed_increment = self.accel*self.trueDT #Find change in speed from current speed
        self.speed = self.prev_speed + speed_increment #Update speed with speed increment

        if (self.speed >= self.max_speed): #Speed regulator for safety
            self.speed = self.max_speed
        if (self.speed <= self.min_speed): #Prevent stalling
            self.speed = self.min_speed

        self.prev_speed = self.speed #Previous speed now holds value of current speed for the next iteration to happen

    def state_forward(self, x_vec): #Function to change current state of car based on control inputs; uses kinematic bicycle model

        beta = np.arctan((self.L_R/(self.L_F+self.L_R))*np.tan(self.steer))

        x_dot = x_vec[3]*np.cos(x_vec[2]+ beta) #Rate of change of longitudinal position
        y_dot = x_vec[3]*np.sin(x_vec[2]+beta) #Rate of change of lateral position
        headAng_dot = (x_vec[3]*np.cos(beta)*np.tan(self.steer))/(self.L_R + self.L_F) # Rate of change of heading

        x_new = []
        x_new.append(x_vec[0] + x_dot*self.trueDT) #New longitudinal position
        x_new.append(x_vec[1] + y_dot*self.trueDT) #New lateral position
        x_new.append(x_vec[2] + headAng_dot*self.trueDT) #New heading angle
        x_new.append(self.speed) #New longitudinal velocity
        x_new.append(0) #Lateral velocity is always zero in a kinematic bicycle model
        x_new.append(self.steer) #New steer angle

        return x_new
    
    def generate_steerdot_control_sequence(self, mean, variance): #Generate random steering rate control inputs with normal distribution

        u_steerdot = torch.normal(mean, variance, size =(self.HORIZON_LENGTH, self.NUM_TRAJECTORIES) )
        return u_steerdot #Has HORIZON_LENGTH rows, NUM_TRAJECTORIES columns, Gaussian distribution

    def generate_throttle_control_sequence(self, mean, variance): #Generate random acceleration control inputs with a normal distribution
    
        u_throttle = torch.normal(mean, variance, size=(self.HORIZON_LENGTH, self.NUM_TRAJECTORIES) )
        return u_throttle #Has HORIZON_LENGTH rows, NUM_TRAJECTORIES columns, Gaussian distribution

    def lidar_callback(self, msg: LaserScan): #Process in the lidar measurements
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

        self.lidar_ranges = np.array(self.lidar_ranges)
        self.lidar_ranges = self.lidar_ranges[:-180] #Remove first and last 180 points to make lidar "see" only from -pi/2 to pi/2 angles
        self.lidar_ranges = self.lidar_ranges[180:]
        self.angle_min = -1.57079632679
        self.angle_max = 1.57079632679

    def compute_centerline(self): #From lidar, extract a centerline
                            
        num_points = int(np.round((self.angle_max - self.angle_min)/self.angle_increment)) #Total number of LiDAR points reported by the LiDAR
        raw_centerline = np.zeros([int(num_points/2), 2]) #Each row is an (x, y) point representing the centerline. x is ahead of car, y is side to side

        for i in range(int(num_points/2)):
            angle = self.angle_min + i*self.angle_increment #angle if car is facing straight down the middle of the track
            x_of_left_point = self.lidar_ranges[i]*np.sin(1.5708 + angle) #Relative x position of point on left wall with respect to LiDAR
            y_of_left_point = self.lidar_ranges[i]*np.cos(1.5708 + angle) #Relative y position (ahead of car) of point on left wall with respect to LiDAR

            x_of_right_point = self.lidar_ranges[num_points- 1 - i]*np.cos(-1*angle) #Relative x position of point on right wall with respect to LiDAR
            y_of_right_point = self.lidar_ranges[num_points - 1 - i]*np.sin(angle) #Relative y position of point on right wall with respect to LiDAR

            x_of_mid_point = (x_of_right_point + x_of_left_point)/2 #Average of x-coordinates for longitudinal distance away from car
            y_of_mid_point = (y_of_left_point + y_of_right_point)/2 #Average of y-coordinates for lateral distance away from car

            raw_centerline[i,0] = x_of_mid_point
            raw_centerline[i,1] = y_of_mid_point

        #Centerline results are less valid at higher heading angles.
        #raw_centerline is nx2, n = (number of LiDAR pts/2). At low n, the centerline points are closer to the car, and at high n, they are further away.
            
        div_factor = np.floor((num_points/2)/self.NUM_RACELINE_PTS) #Corresponds to how far apart points from the raw centerline are taken to form the reduced centerline
        self.centerline = np.empty([self.NUM_RACELINE_PTS, 2]) #Create an array of (x, y) points to store the reduced centerline
        for i in range(self.NUM_RACELINE_PTS): 
            self.centerline[i] = raw_centerline[int(i*div_factor)] #The reduced centerline is the raw centerline at intervals

        index_to_break = self.NUM_RACELINE_PTS - 1 #Index at which reduced centerline is cut short due to measurement noise
        for i in range(self.NUM_RACELINE_PTS - 1): #Loop through the reduced centerline
            dist = np.sqrt((self.centerline[i+1, 0] - self.centerline[i, 0])**2 + (self.centerline[i+1, 1] - self.centerline[i, 1])**2) #distance between consecutive points
            if (dist >= (self.L_R + self.L_F) and i>=3): #If distance between centerline points is more than the length of the car, break the reduced centerline there
                index_to_break = i
                break
        self.centerline = self.centerline[:(index_to_break+2)] #Include a couple centerline points from beyond the breakpoint because these can be helpful for heading

        self.centerline_x = torch.FloatTensor(self.centerline[:,0])
        self.centerline_y = torch.FloatTensor(self.centerline[:,1])
        self.centerline_to_publish = np.concatenate((self.centerline_x, self.centerline_y)) #For publishing the centerline in the visualization

    def find_indices_of_smallest_cost(self, cost_tensor): #Find the indices corresponding to the ten lowest-cost trajectories 
        max_cost = torch.max(cost_tensor)
        optimal_indices = np.empty(10)
        for i in range(self.num_cost_indices):
            min_index = torch.argmin(cost_tensor)
            optimal_indices[i] = min_index #Save the index of minimum cost
            cost_tensor[min_index] = max_cost #If have already found a minimum cost, set it to the maximum cost so it's invisible to argmin

        return optimal_indices #Array with lowest cost trajectories, in order from the lowest cost to the tenth lowest cost trajectory


def main(args=None):
    rclpy.init()
    mppi = MPPIControl() #Initialize node mppi
    rclpy.spin_once(mppi)
    mppi.publish_ackermann() #Zero speed, zero steering angle
    
    x_vec = [0,0,0,0,0,0]
    flag = True
    #for i in range(1000): #This line and the commented lines at the end of the loop may be used for speed testing
    while(flag):
        start_time = time.time() #Begin logging time

        rclpy.spin_once(mppi) #Press Ctrl + C in terminal to terminate node

        x = np.full([mppi.NUM_TRAJECTORIES, mppi.DIM_STATE_VEC], x_vec) # Current state of car
        mppi.x = torch.tensor(np.transpose(x)).float()

        mppi.cost = torch.FloatTensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Initialize cost
        mppi.u_steerdot = mppi.generate_steerdot_control_sequence(0, 2*3.141/6) #rad/s, (mean, variance)
        mppi.u_throttle = mppi.generate_throttle_control_sequence(0,1) #m/s^2, (mean, variance)

        ## Send all states/control inputs/cost array to the device
        mppi.state_x_cuda = torch.FloatTensor(mppi.x[0,:]).cuda()
        mppi.state_y_cuda = torch.FloatTensor(mppi.x[1,:]).cuda()
        mppi.state_psi_cuda = torch.FloatTensor(mppi.x[2,:]).cuda()
        mppi.state_vx_cuda = torch.FloatTensor(mppi.x[3,:]).cuda()
        mppi.state_vy_cuda = torch.FloatTensor(mppi.x[4,:]).cuda()
        mppi.state_steerang_cuda = torch.FloatTensor(mppi.x[5,:]).cuda()
        mppi.cost_cuda = mppi.cost.cuda()
        mppi.flattened_u_steerdot = mppi.u_steerdot.flatten().float().cuda() #Might not need .float()
        mppi.flattened_u_throttle = mppi.u_throttle.flatten().float().cuda() #Might not need .float()

        mppi.compute_centerline() #Update centerline before forwarding dynamics
        mppi.x_centerline = mppi.centerline_x.cuda()
        mppi.y_centerline = mppi.centerline_y.cuda()

        #Do dynamics forwarding and cost computation using CUDA:
        output = time_horizon_cuda.time_horizon_rollout(mppi.state_x_cuda, mppi.state_y_cuda, mppi.state_psi_cuda, mppi.state_vx_cuda, mppi.state_vy_cuda, mppi.state_steerang_cuda, mppi.flattened_u_steerdot, mppi.flattened_u_throttle, mppi.prev_steer_cmd, mppi.prev_accel_cmd, mppi.POS_COST_SCALER, mppi.HEADING_COST_SCALER, mppi.CTRL_INPUT_COST_SCALER, mppi.VEL_COST_SCALER, mppi.V_DESIRED, mppi.x_centerline, mppi.y_centerline, mppi.cost_cuda)
        #All array inputs are 1D Float Tensors sent to cuda, prev_cmds are floats
        #Output is a python list
        output = output[0]
        mppi.cost = torch.tensor(output).clone().detach()

        mppi.optimal_index = mppi.find_indices_of_smallest_cost(mppi.cost) #Find ten lowest-cost trajectories for the visualization
        index_optimal_trajectory = torch.argmin(mppi.cost) #Select trajectory with the minimal cost

        steerdot_command_pub = mppi.u_steerdot[0,index_optimal_trajectory].item() #Pop the control inputs associated with the lowest-cost trajectory
        throttle_command_pub = mppi.u_throttle[0, index_optimal_trajectory].item()

        mppi.set_steer(steerdot_command_pub) #Update steering and speed from the optimal steering rate and accel
        mppi.set_speed(throttle_command_pub)

        mppi.steercmds_to_publish = mppi.u_steerdot #Prep the full randomly generated control input arrays for the visualization
        mppi.throttlecmds_to_publish = mppi.u_throttle

        mppi.publish_ackermann() #Actuate

        mppi.publish_centerline() #Send the centerline to the visualization
        mppi.publish_steercmds() #Send the steering rate commands to the visualization
        mppi.publish_throttlecmds() #Send the acceleration commands to the visualization
        mppi.publish_optimal_traj_index() #Send the trajectory indices of then ten lowest-cost trajectories to the visualization
        mppi.publish_state_vec() #Send the state vector to the visualization

        mppi.prev_steer_cmd = steerdot_command_pub #Acknowledge that a time step has passed
        mppi.prev_accel_cmd = throttle_command_pub

        x_vec = mppi.state_forward(x_vec) #Forward the state in time given the control commands
        x_vec[0] = 0 #This and the next two lines reflect the fact that the LIDAR always comes from the origin (the car)
        x_vec[1] = 0
        x_vec[2] = 0

        end_time = time.time()
        mppi.trueDT = end_time - start_time #Record time for one iteration to calculate how much to change speed/steering angle
        #timeVec.append(end_time - start_time)
    #print(sum(timeVec)/len(timeVec))

if __name__ == '__main__':
    main()
