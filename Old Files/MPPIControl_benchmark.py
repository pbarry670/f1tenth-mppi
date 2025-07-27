#!usr/bin/env python3
# MPPIControl.py without ros related code

import numpy as np
from scipy import interpolate
import torch
import time

from ExecutionTimer import ExecutionTimer

device = torch.device('cpu')

class MPPIControl():

    def __init__(self):

        self.t = ExecutionTimer(True)

        self.steer = 0.0 #Starting steering angle is zero
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


        #Algorithm in a Loop:
            #forward_dynamics using x_cuda, u_steering_cuda, u_throttle_cuda
            #While forwarding dynamics, run compute_cost on cost_cuda. Get lidar measurements and use the centerline to compute position and heading cost
            #Get the index of the best trajectory using evaluate_best_cost, this is stored in self.index
            #Obtain the control commands for this trajectory
            #Actuate and reset states, cost
            #Restart the control loop, generating a new warm-started set of random control commands. Send cost back to cuda.
    

    def set_steer(self, steer_cmd):
        self.steer = steer_cmd

    def set_accel(self, accel_cmd):
        self.accel = accel_cmd

    def forward_dynamics(self, x_cuda, u_steering_cuda, u_throttle_cuda): #Function to forward the dynamics by one time step; variables already in CUDA should be passed in

        beta = torch.atan((self.L_R/(self.L_F+self.L_R))*torch.tan(u_steering_cuda))

        x_dot = x_cuda[3,:] * torch.cos(x_cuda[2,:] + beta)
        y_dot = x_cuda[3,:] * torch.sin(x_cuda[2,:] + beta)
        headAng_dot = (x_cuda[3,:]*torch.cos(beta)*torch.tan(u_steering_cuda))/(self.L_R + self.L_F)
        x_cuda[0,:] = x_cuda[0,:] + x_dot*self.DT #Update x position of car
        x_cuda[1,:] = x_cuda[1,:] + y_dot*self.DT #Update y position of car
        x_cuda[2,:] = x_cuda[2,:] + headAng_dot*self.DT #Update heading angle of car
        x_cuda[3,:] = x_cuda[3,:]+ self.throttle_to_accel_mapping(u_throttle_cuda)*self.DT

        return x_cuda

    def state_forward(self, x_vec, u_steer, u_throttle): #Dummy function for testing

        beta = np.arctan((self.L_R/(self.L_F+self.L_R))*np.tan(u_steer))

        x_dot = x_vec[3]*np.cos(x_vec[2]+ beta)
        y_dot = x_vec[3]*np.sin(x_vec[2]+beta)
        headAng_dot = (x_vec[3]*np.cos(beta)*np.tan(u_steer))/(self.L_R + self.L_F)

        x_new = []
        x_new.append(x_vec[0] + x_dot*self.DT)
        x_new.append(x_vec[1] + y_dot*self.DT)
        x_new.append(x_vec[2] + headAng_dot*self.DT)
        x_new.append(x_vec[3] + self.throttle_to_accel_mapping(u_throttle)*self.DT)
        x_new.append(0) #Lateral velocity

        return x_new

    def compute_cost(self, x_cuda, u_steer, u_throttle):

        pos_cost, heading_cost = self.compute_position_and_heading_cost(x_cuda)

        cost = self.VEL_COST_SCALER*torch.square(self.V_DESIRED - x_cuda[3,:]) + self.POS_COST_SCALER*torch.square(pos_cost) + self.HEADING_COST_SCALER*heading_cost + self.CTRL_INPUT_COST_SCALER*(torch.square(u_steer) + torch.square(u_throttle))
        return cost

    #Cost for having command be too far from previous command (both steering and throttle). Big cost for commands out of bounds
        #Or, can just create a cost for the magnitude of the control input. This is what's implemented right now.
    #Cost on deviation from raceline
    #Cost on heading
    #Cost on difference between V_desired and v

    def compute_position_and_heading_cost(self, x_cuda):
#x_cuda = [x, y, heading, long_v, lat_v)
        y_loc = self.interp(self.centerline_x_cuda, self.centerline_y_cuda, x_cuda[0,:]) #Obtain y location based on spline interpolation
        y_dist = y_loc - x_cuda[1,:]
        pos_cost = (torch.abs(y_dist))/self.RACELINE_WIDTH

        optimal_heading = torch.atan2(self.centerline_dy_cuda, self.centerline_dx_cuda)
        interpolated_heading = self.interp(self.centerline_x_short_cuda, optimal_heading, x_cuda[0,:])
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
        self.centerline = np.array([[0.03,0],[0.08,0.0],[0.15,0.0],[0.23,0.0],[0.31,0.0],[0.42,0.0],[0.52,0.0],[0.58,0],[0.66,0],[0.78,0.0],[0.84,0.0],[0.91,0.0],[0.99,0.0],[1.08,0.0],[1.17,0.0],[1.26,0.0],[1.34,0.0],[1.56,0.0],[1.63,0.0],[1.9,0.0],[2.2,0.0]])
        # THE LINE ABOVE IS FOR TESTING ONLY #

        centerline_x = self.centerline[:,0]
        centerline_y = self.centerline[:,1]
        last_index = len(centerline_x) - 1

        centerline_x[0:16] = np.sort(centerline_x[0:16])
        for i in range(len(centerline_x) - 1):
            if centerline_x[i+1] < centerline_x[i]:
                last_index = i
                break

        centerline_dx = centerline_x[1:-1] - centerline_x[0:-2]
        centerline_dy = centerline_y[1:-1] - centerline_y[0:-2]
        centerline_x_short = torch.tensor(centerline_x[0:-2])

        centerline_x = torch.tensor(centerline_x)
        centerline_y = torch.tensor(centerline_y)
        centerline_dx = torch.tensor(centerline_dx)
        centerline_dy = torch.tensor(centerline_dy)

        self.centerline_x_cuda = centerline_x.to(device)
        self.centerline_y_cuda = centerline_y.to(device)
        self.centerline_dx_cuda = centerline_dx.to(device)
        self.centerline_dy_cuda = centerline_dy.to(device)
        self.centerline_x_short_cuda = centerline_x_short.to(device)

    def h_poly_helper(self, tt):
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
            ], dtype=tt[-1].dtype)
        return [
            sum( A[i, j]*tt[j] for j in range(4) )
                for i in range(4) ]

    def h_poly(self, t):
        tt = [ None for _ in range(4) ]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i-1]*t
        return self.h_poly_helper(tt)

    def H_poly(self, t):
        tt = [ None for _ in range(4) ]
        tt[0] = t
        for i in range(1, 4):
            tt[i] = tt[i-1]*t*i/(i+1)
        return self.h_poly_helper(tt)

    def interp_func(self, x, y):
        "Returns integral of interpolating function"
        if len(y)>1:
            m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
            m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
        def f(xs):
            if len(y)==1: # in the case of 1 point, treat as constant function
                return y[0] + torch.zeros_like(xs)
            I = torch.searchsorted(x[1:], xs)
            dx = (x[I+1]-x[I])
            hh = self.h_poly((xs-x[I])/dx)
            return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
        return f

    def interp(self, x, y, xs):
        return self.interp_func(x,y)(xs)

    def dummy_lidar_callback(self):
        '''create a dummy lidar scan with car in the middle of a straight corridor'''
        N = 200
        width = 0.4
        # straight corridor

        theta_vec = np.linspace(0,np.pi,N//2)
        left_scan = width/2 / np.sin(theta_vec)
        theta_vec = np.linspace(np.pi,np.pi*2,N//2)
        right_scan = -width/2 / np.sin(theta_vec)

        self.lidar_ranges = np.hstack([left_scan,right_scan])
        self.angle_min = 0
        self.angle_max = np.pi*2
        self.angle_increment = (self.angle_max - self.angle_min) / (N-1)
   
def main(args=None):
    mppi = MPPIControl() #Initialize node mppi
    mppi.dummy_lidar_callback()

    mppi.u_steer = mppi.generate_throttle_control_sequence(0, 3.141/6)
    
    mppi.u_throttle = mppi.generate_throttle_control_sequence(0,1) #mean of zero, variance of 1
    
    mppi.x0 = torch.tensor(np.zeros([mppi.DIM_STATE_VEC, mppi.NUM_TRAJECTORIES])) #Initial state vector array with all trajectories
    
    mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Initialize cost

    start_time = time.time()

    x0_cuda = mppi.x0.to(device)
    u_steer_cuda = mppi.u_steer.to(device)
    u_throttle_cuda = mppi.u_throttle.to(device)
    cost_cuda = mppi.cost.to(device)

    mppi.compute_centerline() #Update centerline before forwarding dynamics

    #Start the run from zero initial conditions
    for i in range(mppi.HORIZON_LENGTH - 1): #Loop through the dynamics with each sequential control input over the horizon
        x_new = mppi.forward_dynamics(x0_cuda, u_steer_cuda[i,:], u_throttle_cuda[i,:])
        cost_cuda = cost_cuda + mppi.compute_cost(x_new, u_steer_cuda[i,:], u_throttle_cuda[i,:]) #Compute cost at each step forward in time
    
    mppi.cost = cost_cuda.cpu()
    index_optimal_trajectory = torch.argmin(mppi.cost) #Select trajectory with the minimal cost
    steer_command = u_steer_cuda[0,index_optimal_trajectory]
    throttle_command = u_throttle_cuda[0, index_optimal_trajectory]

    mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset costs to zero for next projection forward in time

    steer_command = steer_command.cpu()
    throttle_command = throttle_command.cpu()

    steer_command_pub = steer_command.item() #Values to be published to Ackermann
    throttle_command_pub = throttle_command.item()

    mppi.set_steer(steer_command_pub)
    mppi.set_accel(throttle_command_pub)
    #mppi.publish_ackermann()

    x_vec = [0,0,0,0,0]
    x_vec = mppi.state_forward(x_vec, steer_command_pub, throttle_command_pub) #Testing dummy for state estimate


    end_time = time.time()
    print(f"Time taken for initialization: {end_time - start_time}")

    x_vec[0] = 0 #Update x location to reflect the fact the origin is centered on car
    flag = True

    #while(flag):
    # simulate for 100 steps
    for i in range(100):
        mppi.t.s()

        x = np.full([mppi.NUM_TRAJECTORIES, mppi.DIM_STATE_VEC], x_vec)
        mppi.x = torch.tensor(np.transpose(x))

        mppi.t.s('generate control seq')
        mppi.u_steer = mppi.generate_steering_control_sequence(steer_command_pub, 3.141/6)
        mppi.u_throttle = mppi.generate_throttle_control_sequence(throttle_command_pub, 1)
        mppi.t.e('generate control seq')

        mppi.t.s('compute_centerline')
        mppi.compute_centerline() #generate nx2 array -- (x, y) -- of raceline points here (the center of the track) from LIDAR
        mppi.t.e('compute_centerline')

        mppi.t.s('to_cuda')
        x_cuda = mppi.x.to(device)
        u_steer_cuda = mppi.u_steer.to(device)
        u_throttle_cuda = mppi.u_throttle.to(device)
        cost_cuda = mppi.cost.to(device)
        mppi.t.e('to_cuda')

        mppi.t.s('rollout')
        for i in range(mppi.HORIZON_LENGTH - 1):
            x_new = mppi.forward_dynamics(x_cuda, u_steer_cuda[i,:], u_throttle_cuda[i,:])
            cost_cuda = cost_cuda + mppi.compute_cost(x_new, u_steer_cuda[i,:], u_throttle_cuda[i,:])

        mppi.cost = cost_cuda.cpu()
        mppi.t.e('rollout')

        mppi.t.s('synthesize control')
        index_optimal_trajectory = torch.argmin(mppi.cost)
        steer_command = u_steer_cuda[0, index_optimal_trajectory].cpu()
        throttle_command = u_throttle_cuda[0, index_optimal_trajectory].cpu()

        steer_command_pub = steer_command.item() #Values to be published to Ackermann
        throttle_command_pub = throttle_command.item()
        mppi.t.e('synthesize control')

        mppi.set_steer(steer_command_pub)
        mppi.set_accel(throttle_command_pub)
        #mppi.publish_ackermann()

        mppi.cost = torch.tensor(np.zeros(mppi.NUM_TRAJECTORIES)) #Reset Cost
        x_vec = mppi.state_forward(x_vec, steer_command_pub, throttle_command_pub) #State estimation for testing
        x_vec[0] = 0 #This and the next line reflect the fact that the LIDAR always comes from the origin
        print(x_vec)
        #print(f"Throttle Command: {throttle_command_pub}")
        #print(f"Steering Command: {steer_command_pub}")
        end_time = time.time()
        mppi.t.e()
    mppi.t.summary()




if __name__ == '__main__':
    main()
