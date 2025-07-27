#!usr/bin/env python3

import numpy as np
from scipy import interpolate
import torch
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
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
        self.ack_publisher = self.create_publisher(AckermannDriveStamped, "/ackermann_cmd", qos_profile=ack_qos_profile) #Publish AckermannDriveStamped messages to /drive topic
        #self.timer = self.create_timer(0.01, self.publish_ackermann) #Publish ackermann commands every 0.025 seconds
        self.centerline_publisher = self.create_publisher(Float32MultiArray, "/centerline", 10)
        self.steering_publisher = self.create_publisher(Float32MultiArray, "/steercmds", 10)
        self.throttle_publisher = self.create_publisher(Float32MultiArray, "/throttlecmds", 10)
        self.optimal_traj_index_publisher = self.create_publisher(Float32MultiArray, "/optimal_index", 10)
        self.state_publisher = self.create_publisher(Float32MultiArray, "/state_vec", 10)

        self.cost_params_sub = self.create_subscription(Float32MultiArray, "/cost_params", self.cost_params_callback, 10)
        self.cost_params_sub #Avoid unused variable warning

        self.steer = 0.0 #Starting steering angle is zero
        self.accel = 0.0 #Starting acceleration is zero
        self.speed = 0.0 #Starting speed is zero
        self.HORIZON_LENGTH = 20 #Number of time steps over which trajectories are forwarded
        self.NUM_TRAJECTORIES = 1000 #Number of trajectories
        self.DIM_STATE_VEC = 5 #x, y, headAng, lateral V, longitudinal V

        self.prev_steer_cmd = 0.0
        self.prev_accel_cmd = 0.0
        self.prev_speed = 0.0
        self.trueDT = 0.048 #Test time step; should correspond to time step between actuation

        self.L_F = 0.15 #Front axle to cg, m
        self.L_R = 0.15 #Rear axle to cg, m
        self.NUM_RACELINE_PTS = 60

        self.list_of_indices = []
        for i in range(self.HORIZON_LENGTH):
            self.list_of_indices.append(i)
            
        self.list_of_indices = np.array(self.list_of_indices)
        self.num_cost_indices = 10
        self.opimal_index = np.tile(1, self.num_cost_indices)


        #Initialize cost parameters, though these will be changed via a subscription to a topic with the real values used
        self.V_DESIRED = 2.0 #desired car velocity, m/s
        self.VEL_COST_SCALER = 1.0 #scalar parameter to tune cost associated with deviating from velocity cost
        #self.RACELINE_WIDTH = 1 #distance from track boundary to track boundary, m
        #self.NUM_RACELINE_PTS = 30 #Number of points on discretized centerline
        self.POS_COST_SCALER = 1.0 #scalar parameter to tune cost associated with deviation from centerline
        self.HEADING_COST_SCALER = 10.0 #scalar parameter to tune cost associated with devition from ideal heading angle
        self.CTRL_INPUT_COST_SCALER = 0.1 #scalar parameter to tune cost associated with magnitude of control inputs
        #Algorithm in a Loop:
            #forward_dynamics using x_cuda, u_steering_cuda, u_throttle_cuda
            #While forwarding dynamics, run compute_cost on cost_cuda. Get lidar measurements and use the centerline to compute position and heading cost
            #Get the index of the best trajectory using evaluate_best_cost, this is stored in self.index
            #Obtain the control commands for this trajectory
            #Actuate and reset states, cost
            #Restart the control loop, generating a new warm-started set of random control commands. Send cost back to cuda.
    
    def publish_ackermann(self):
        ack_msg = AckermannDriveStamped() #ROS2 msg type
        ack_msg.header.stamp = rclpy.time.Time().to_msg() #timestamp
        ack_msg.header.frame_id = "laser" #frame ID of message
        ack_msg.drive.steering_angle= self.steer #steer command, radians
        ack_msg.drive.speed = self.speed
        #ack_msg.drive.acceleration = self.accel #accel command, m/s^2
        self.ack_publisher.publish(ack_msg) #publish AckermannDriveStamped message to /ackermann_cmd

    def publish_centerline(self):
        centerline_msg = Float32MultiArray()
        self.centerline_to_publish = Float32MultiArray(data=self.centerline_to_publish)
        self.centerline_publisher.publish(self.centerline_to_publish)

    def publish_steercmds(self):
        steer_msg = Float32MultiArray()
        #self.steercmds_to_publish is a self.HORIZON_LENGTH x self.NUM_TRAJECTORIES array
        self.intermediate_steercmds = np.empty((self.HORIZON_LENGTH*self.NUM_TRAJECTORIES))
        for i in range(self.NUM_TRAJECTORIES):
            self.intermediate_steercmds[i*self.HORIZON_LENGTH + self.list_of_indices] = self.steercmds_to_publish[:, i]

        self.steercmds_to_publish = Float32MultiArray(data=self.intermediate_steercmds)
        self.steering_publisher.publish(self.steercmds_to_publish)

    def publish_throttlecmds(self):
        throttle_msg = Float32MultiArray()

        self.intermediate_throttlecmds = np.empty((self.HORIZON_LENGTH*self.NUM_TRAJECTORIES))
        for i in range(self.NUM_TRAJECTORIES):
            self.intermediate_throttlecmds[i*self.HORIZON_LENGTH + self.list_of_indices] = self.throttlecmds_to_publish[:,i]

        self.throttlecmds_to_publish = Float32MultiArray(data=self.intermediate_throttlecmds)
        self.throttle_publisher.publish(self.throttlecmds_to_publish)

    def publish_optimal_traj_index(self):
        index_msg = Float32MultiArray()
        self.index_to_publish = Float32MultiArray(data=self.optimal_index)
        self.optimal_traj_index_publisher.publish(self.index_to_publish)
        
    def publish_state_vec(self):
        state_msg = Float32MultiArray()
        self.state = [0.0, 0.0, 0.0, self.speed, 0.0] #[x, y, psi, vx, vy]
        self.state_to_publish = Float32MultiArray(data=self.state)
        self.state_publisher.publish(self.state_to_publish)

    def cost_params_callback(self, msg : Float32MultiArray):
        self.cost_params = msg.data
        self.V_DESIRED = self.cost_params[0]
        self.VEL_COST_SCALER = self.cost_params[1]
        self.POS_COST_SCALER = self.cost_params[2]
        self.HEADING_COST_SCALER = self.cost_params[3]
        self.CTRL_INPUT_COST_SCALER = self.cost_params[4]

    def set_steer(self, steer_cmd):
        self.steer = steer_cmd

    def set_accel(self, accel_cmd):
        self.accel = accel_cmd
        speed_increment = self.accel*self.trueDT
        self.set_speed(speed_increment)

    def set_speed(self, speed_increment):
        self.speed = self.prev_speed + speed_increment
        print(self.speed)
        if (self.speed >= 2.0): # Speed regulator for safety
            self.speed = 2.0
        self.prev_speed = self.speed

    def state_forward(self, x_vec, u_steer, u_throttle): #Dummy function for testing

        beta = np.arctan((self.L_R/(self.L_F+self.L_R))*np.tan(u_steer))

        x_dot = x_vec[3]*np.cos(x_vec[2]+ beta)
        y_dot = x_vec[3]*np.sin(x_vec[2]+beta)
        headAng_dot = (x_vec[3]*np.cos(beta)*np.tan(u_steer))/(self.L_R + self.L_F)

        x_new = []
        x_new.append(x_vec[0] + x_dot*self.trueDT)
        x_new.append(x_vec[1] + y_dot*self.trueDT)
        x_new.append(x_vec[2] + headAng_dot*self.trueDT)
        x_new.append(self.speed)
        x_new.append(0) #Lateral velocity

        return x_new
    
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

    def lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = msg.ranges #Array with ranges starting from angle_min ranging to angle_max
        self.angle_min = msg.angle_min #minimum angle at which LiDAR records
        self.angle_max = msg.angle_max #maximum angle at which LiDAR records
        self.angle_increment = msg.angle_increment #how much angular distance is between two points on the LiDAR

        self.lidar_ranges = np.array(self.lidar_ranges)
        self.lidar_ranges = self.lidar_ranges[:-180] #Remove first and last 180 points to make lidar "see" only from -pi/2 to pi/2 angles
        self.lidar_ranges = self.lidar_ranges[180:]
        self.angle_min = -1.57079632679
        self.angle_max = 1.57079632679

    def compute_centerline(self):
                            
        num_points = int(np.round((self.angle_max - self.angle_min)/self.angle_increment)) #720
        
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


        #Note: centerline results are less valid at higher heading angles.
        #Note: raw_centerline is nx2, n = (number of LiDAR pts/2). At low n, the centerline points are closer to the car, and at high n, they are further away.
        div_factor = np.floor((num_points/2)/self.NUM_RACELINE_PTS)
        uncertainty_factor = 6 #Number of NUM_RACELINE_PTS that we choose to not trust (are cut off at the end of the computed centerline). Curvier/narrower track --> more uncertainty
        self.centerline = np.empty([self.NUM_RACELINE_PTS - uncertainty_factor, 2])
        for i in range(self.NUM_RACELINE_PTS - uncertainty_factor): 
            self.centerline[i] = raw_centerline[int(i*div_factor)]

        

        # THE LINE BELOW IS FOR TESTING ONLY #
        #self.centerline = np.array([[0.03,0],[0.08,0.0],[0.15,0.0],[0.23,0.0],[0.31,0.0],[0.42,0.0],[0.52,0.0],[0.58,0],[0.66,0],[0.78,0.0],[0.84,0.0],[0.91,0.0],[0.99,0.0],[1.08,0.0],[1.17,0.0],[1.26,0.0],[1.34,0.0],[1.56,0.0],[1.63,0.0],[1.9,0.0],[2.2,0.0]])
        # THE LINE ABOVE IS FOR TESTING ONLY #

        centerline_x = self.centerline[:,0]
        centerline_y = self.centerline[:,1]
        removable_indices = []

        for i in range(len(centerline_x) - 1):
            if(centerline_x[i] <= 0.0000001):
                removable_indices.append(i)

        centerline_x = np.delete(centerline_x, removable_indices)
        centerline_y = np.delete(centerline_y, removable_indices)

        centerline_size = len(centerline_x)

        self.centerline_x = torch.FloatTensor(centerline_x)
        self.centerline_y = torch.FloatTensor(centerline_y)
        #self.centerline_x = self.centerline_x[0:(centerline_size-1)]
        #self.centerline_y = self.centerline_y[0:(centerline_size-1)]
        self.centerline_to_publish = np.concatenate((self.centerline_x, self.centerline_y))

    def find_indices_of_smallest_cost(self, cost_tensor):
        # find self.num_cost_indices indices that have the lowest 10 values in the cost function
        max_cost = torch.max(cost_tensor)
        optimal_indices = np.empty(10)
        for i in range(self.num_cost_indices):
            min_index = torch.argmin(cost_tensor)
            optimal_indices[i] = min_index
            cost_tensor[min_index] = max_cost

        return optimal_indices


def main(args=None):
    rclpy.init()
    mppi = MPPIControl() #Initialize node mppi
    mppi.publish_ackermann() #Zero speed, steering
    rclpy.spin_once(mppi) #Press Ctrl + C in terminal to terminate node

    mppi.u_steer = mppi.generate_throttle_control_sequence(0, 3.141/6)
    
    mppi.u_throttle = mppi.generate_throttle_control_sequence(1,1) #mean of 1 m/s^2, variance of 1
    
    mppi.x0 = np.zeros([mppi.DIM_STATE_VEC, mppi.NUM_TRAJECTORIES]) #Initial state vector array with all trajectories
    
    mppi.cost = torch.FloatTensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Initialize cost

    start_time = time.time()

    mppi.state_x_cuda = torch.FloatTensor(mppi.x0[0,:]).cuda()
    mppi.state_y_cuda = torch.FloatTensor(mppi.x0[1,:]).cuda()
    mppi.state_psi_cuda = torch.FloatTensor(mppi.x0[2,:]).cuda()
    mppi.state_vx_cuda = torch.FloatTensor(mppi.x0[3,:]).cuda()
    mppi.state_vy_cuda = torch.FloatTensor(mppi.x0[4,:]).cuda()
    mppi.cost_cuda = mppi.cost.cuda()
    mppi.flattened_u_steer = mppi.u_steer.flatten().float().cuda() #Might not need .float()
    print(mppi.flattened_u_steer)
    mppi.flattened_u_throttle = mppi.u_throttle.flatten().float().cuda() #Might not need .float()

    mppi.compute_centerline() #Update centerline before forwarding dynamics
    mppi.x_centerline = mppi.centerline_x.cuda()
    mppi.y_centerline = mppi.centerline_y.cuda()
    # mppi.prev_steer_cmd, mppi.prev_accel_cmd are also sent to the kernel

    #Send the dynamics and the control inputs over the time horizon as well as computing cost
    output = time_horizon_cuda.time_horizon_rollout(mppi.state_x_cuda, mppi.state_y_cuda, mppi.state_psi_cuda, mppi.state_vx_cuda, mppi.state_vy_cuda, mppi.flattened_u_steer, mppi.flattened_u_throttle, mppi.prev_steer_cmd, mppi.prev_accel_cmd, mppi.POS_COST_SCALER, mppi.HEADING_COST_SCALER, mppi.CTRL_INPUT_COST_SCALER, mppi.VEL_COST_SCALER, mppi.V_DESIRED, mppi.x_centerline, mppi.y_centerline, mppi.cost_cuda)
    #All array inputs are 1D Float Tensors sent to cuda, prev_cmds are floats
    #Output is a python list
    output = output[0]
    mppi.cost = torch.tensor(output).clone().detach()

    mppi.optimal_index = mppi.find_indices_of_smallest_cost(mppi.cost)
    index_optimal_trajectory = torch.argmin(mppi.cost) #Select trajectory with the minimal cost

    steer_command_pub = mppi.u_steer[0,index_optimal_trajectory].item() #Pop the control inputs associated with the lowest-cost trajectory
    throttle_command_pub = mppi.u_throttle[0, index_optimal_trajectory].item()

    if (steer_command_pub >= 0.4):
        steer_command_pub = 0.4
    if (steer_command_pub <= -0.4):
        steer_command_pub = -0.4

    mppi.cost = torch.FloatTensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset costs to zero for next projection forward in time

    mppi.set_steer(steer_command_pub)
    mppi.set_accel(throttle_command_pub)

    mppi.steercmds_to_publish = mppi.u_steer
    mppi.throttlecmds_to_publish = mppi.u_throttle

    #mppi.publish_ackermann()

    mppi.publish_centerline()
    mppi.publish_steercmds()
    mppi.publish_throttlecmds()
    mppi.publish_optimal_traj_index()
    mppi.publish_state_vec()

    mppi.prev_steer_cmd = steer_command_pub
    mppi.prev_accel_cmd = throttle_command_pub

    x_vec = [0,0,0,0,0]
    x_vec = mppi.state_forward(x_vec, steer_command_pub, throttle_command_pub) #Testing dummy for state estimate


    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    x_vec[0] = 0 #Update state vec to reflect the fact that origin is centered on car
    x_vec[1] = 0
    x_vec[2] = 0
    flag = True
    timeVec = []

    while(flag):
    #for i in range(1000):
        start_time = time.time()
        rclpy.spin_once(mppi)
        x = np.full([mppi.NUM_TRAJECTORIES, mppi.DIM_STATE_VEC], x_vec)
        mppi.x = torch.tensor(np.transpose(x)).float()

        mppi.u_steer = mppi.generate_steering_control_sequence(0, 3.141/6)
        mppi.u_throttle = mppi.generate_throttle_control_sequence(0, 1)

        mppi.state_x_cuda = torch.FloatTensor(mppi.x[0,:]).cuda()
        mppi.state_y_cuda = torch.FloatTensor(mppi.x[1,:]).cuda()
        mppi.state_psi_cuda = torch.FloatTensor(mppi.x[2,:]).cuda()
        mppi.state_vx_cuda = torch.FloatTensor(mppi.x[3,:]).cuda()
        mppi.state_vy_cuda = torch.FloatTensor(mppi.x[4,:]).cuda()
        mppi.cost_cuda = mppi.cost.cuda()
        mppi.flattened_u_steer = mppi.u_steer.flatten().float().cuda() #Might not need .float()
        #print(mppi.flattened_u_steer)
        mppi.flattened_u_throttle = mppi.u_throttle.flatten().float().cuda() #Might not need .float()

        mppi.compute_centerline() #generate nx2 array -- (x, y) -- of raceline points here (the center of the track) from LIDAR
        mppi.x_centerline = mppi.centerline_x.cuda()
        mppi.y_centerline = mppi.centerline_y.cuda()

        #Send the dynamics and the control inputs over the time horizon as well as computing cost
        output = time_horizon_cuda.time_horizon_rollout(mppi.state_x_cuda, mppi.state_y_cuda, mppi.state_psi_cuda, mppi.state_vx_cuda, mppi.state_vy_cuda, mppi.flattened_u_steer, mppi.flattened_u_throttle, mppi.prev_steer_cmd, mppi.prev_accel_cmd, mppi.POS_COST_SCALER, mppi.HEADING_COST_SCALER, mppi.CTRL_INPUT_COST_SCALER, mppi.VEL_COST_SCALER, mppi.V_DESIRED, mppi.x_centerline, mppi.y_centerline, mppi.cost_cuda)
        
        #All array inputs are 1D Float Tensors sent to cuda, prev_cmds are floats
        output = output[0]
        mppi.cost = torch.tensor(output).clone().detach()

        mppi.optimal_index = mppi.find_indices_of_smallest_cost(mppi.cost)
        index_optimal_trajectory = torch.argmin(mppi.cost)

        steer_command_pub = mppi.u_steer[0, index_optimal_trajectory].item()
        throttle_command_pub = mppi.u_throttle[0, index_optimal_trajectory].item()

        if (steer_command_pub >= 0.4):
            steer_command_pub = 0.4
        if (steer_command_pub <= -0.4):
            steer_command_pub = -0.4

        mppi.set_steer(steer_command_pub)
        mppi.set_accel(throttle_command_pub)
        #print(f"Steering Command: {steer_command_pub}")
        #print(f"Accel Command: {throttle_command_pub}")

        mppi.steercmds_to_publish = mppi.u_steer
        mppi.throttlecmds_to_publish = mppi.u_throttle

        mppi.publish_centerline()
        mppi.publish_steercmds()
        mppi.publish_throttlecmds()
        mppi.publish_optimal_traj_index()
        mppi.publish_state_vec()

        #mppi.publish_ackermann()

        mppi.prev_steer_cmd = steer_command_pub
        mppi.prev_accel_cmd = throttle_command_pub

        mppi.cost = torch.FloatTensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset Cost
        x_vec = mppi.state_forward(x_vec, steer_command_pub, throttle_command_pub) #State estimation for testing
        #print(x_vec)
        x_vec[0] = 0 #This and the next two lines reflect the fact that the LIDAR always comes from the origin
        x_vec[1] = 0
        x_vec[2] = 0
        end_time = time.time()
        mppi.trueDT = end_time - start_time
        #timeVec.append(end_time - start_time)

    #print(sum(timeVec)/len(timeVec))


if __name__ == '__main__':
    main()
