#!usr/bin/env python3
import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class F1TenthKalmanFilter(Node):
    def __init__(self):

        super().__init__("car_kf")

        self.receive_controls = self.create_subscription(Float32MultiArray, "/control_commands", self.control_read_callback, 10)
        self.receive_sensors = self.create_subscription(Float32MultiArray, "/zed_sensor_data", self.sensor_read_callback, 10)
        self.state_publisher = self.create_publisher(Float32MultiArray, "/filtered_state", 10)
        self.covariance_publisher = self.create_publisher(Float32MultiArray, "/covariances", 10)

        self.dt = 0.01 # 100 Hz
        
        ## From Sensor Calibration ##
        self.accel_x_bias = 0.410588
        self.accel_y_bias = -0.19411

        ### Kalman Filter Parameters ###
        self.x = np.zeros(7) # Initial state: [x, y, psi, v, del, a, deldot] 
        self.x_prev = np.zeros(7) # Previous state
        self.P_prev = 0.1*np.eye(7) # Previous covariance
        self.x_prior = np.zeros(7) # Prior state
        self.P_prior = np.zeros((7,7)) # Prior covariance
        self.P_next = np.zeros((7,7))
        self.u = np.zeros(2) # Control vector: [throttle, deldot]
        self.y = np.zeros(3) # Residual, y
        self.h = np.zeros(3)

        self.P = 0.1*np.eye(7) # Initial covariance, 0.1*IdentityMatrix, P
        self.Q = np.eye(7) # Process covariance, Q
        self.Q[0,0] = 0.4
        self.Q[1,1] = 0.4
        self.Q[2,2] = 0.3
        self.Q[3,3] = 0.2
        self.Q[4,4] = 0.2
        self.Q[5,5] = 0.1
        self.Q[6,6] = 0.1
        
        self.R = np.array([[0.4, 0.0, 0.0],
                        [0.0, 0.4, 0.0],
                        [0.0, 0.0, 0.25]]) # Measurement covariance, R
        
        self.F = np.zeros((7,7)) #State transition function, F
        
        self.H = np.zeros((3,7)) # Measurement function, H

        self.K = np.zeros((7,3)) # Kalman gain, K
        
        self.Coriolis_matrix = np.zeros((2,2))
        self.c = np.array([-0.265, 0.0]) # Vector from IMU mounting point to center of mass of car, meters

        self.recent_accel_x = 0.0
        self.recent_accel_y = 0.0
        self.recent_gyro_z = 0.0
        print('Running Kalman filter...')
        
        self.timer = self.create_timer(self.dt, self.run_filter)

    def run_filter(self): #Function that runs during the node execution at a fixed rate
        #print(self.x)
        self.predict_step()
        self.update_step()
        self.publish_state()
        self.publish_covariances()
        self.acknowledge_time_passed()

    def publish_state(self):
        self.state_to_publish = Float32MultiArray(data=self.x)
        self.state_publisher.publish(self.state_to_publish)

    def publish_covariances(self):
        self.covs = [self.P_next[0,0], self.P_next[1,1], self.P_next[2,2]]
        self.covs_to_publish = Float32MultiArray(data=self.covs)
        self.covariance_publisher.publish(self.covs_to_publish)

    def control_read_callback(self, msg : Float32MultiArray):
        self.u = msg.data # u = [a, deldot]
        #self.u = [0.0, 0.0]
        
    def read_controls(self):
        self.a = self.u[0]
        self.deldot = self.u[1]
    
    def sensor_read_callback(self, msg : Float32MultiArray):
        imu_data = msg.data
        self.recent_accel_x = imu_data[2] - self.accel_x_bias
        self.recent_accel_y = -1*(imu_data[0] - self.accel_y_bias)
        self.recent_gyro_z = -1*(3.14159/180)*imu_data[4]

    def read_IMU(self):

        self.z_accel = [self.recent_accel_x, self.recent_accel_y]
        
        #wz = (self.x_prior[3]/0.3302)*np.cos(np.arctan(0.5*np.tan(self.x_prior[4])))*np.tan(self.x_prior[4]) # psi_dot
        
        #self.Coriolis_matrix = np.array([[-1*wz*wz, 0.0],
        #                                [0.0, -1*wz*wz]]) 
            # Derived from https://robotics.stackexchange.com/questions/24276/calculation-of-imu-offset-for-placement-of-inertial-measurement-unit-away-from-c
            # Assumes no wx, no wy, no wxdot, no wydot, and no wzdot
        #self.z_accel = self.z_accel + np.matmul(self.Coriolis_matrix, self.c) # Local reference frame accelerations about CG

        self.Rotation_matrix = np.array([[np.cos(-self.x_prior[2]), -1*np.sin(-self.x_prior[2])],
                                       [np.sin(-self.x_prior[2]), np.cos(-self.x_prior[2])]]) # For rotating local accelerations into global accelerations

        self.z_accel = np.matmul(self.Rotation_matrix, self.z_accel) # Obtain global reference frame accelerations
        self.z = np.append(self.z_accel, self.recent_gyro_z)
       

    def predict_step(self):
        print('Filtered State:\n')
        print(self.x)
        print('\n')
        
        self.compute_F()
        #print('State Transition Matrix:\n')
        #print(self.F)
        #print('\n')
        
        

        self.state_transition_function()
        #print('x_prior before reading controls: \n')
        #print(self.x_prior)
        #print('\n')

        #print('x_prior after reading controls: \n')
        #print(self.x_prior)
        #print('\n')

        self.P_prior = np.matmul(self.F, np.matmul(self.P_prev, np.transpose(self.F))) + self.Q
        #print('P_prior: \n')
        #print(self.P_prior)
        #print('\n')

    def compute_F(self):
        dt = self.dt
        psi = self.x[2]
        delta = self.x[4]
        v = self.x[3]
        a_old = self.a
        deldot_old = self.deldot

        self.F[0,0] = 1
        self.F[0,2] = -dt*v*np.sin(psi+np.arctan(0.5*np.tan(delta)))
        self.F[0,3] = dt*np.cos(psi+np.arctan(0.5*np.tan(delta)))
        self.F[0,4] = -(dt*v*np.sin(psi + np.arctan(0.5000*np.tan(delta)))*(0.5000*np.tan(delta)*np.tan(delta) + 0.5000))/(0.2500*np.tan(delta)*np.tan(delta) + 1)

        self.F[1,1] = 1
        self.F[1,2] = dt*v*np.cos(psi+np.arctan(0.5*np.tan(delta)))
        self.F[1,3] = dt*np.sin(psi+np.arctan(0.5*np.tan(delta)))
        self.F[1,4] = (dt*v*np.cos(psi + np.arctan(0.5000*np.tan(delta)))*(0.5000*np.tan(delta)*np.tan(delta) + 0.5000))/(0.2500*np.tan(delta)*np.tan(delta) + 1)

        self.F[2,2] = 1
        self.F[2,3] = (3.0285*dt*np.tan(delta))/np.sqrt((0.2500*np.tan(delta)^2 + 1))
        self.F[2,4] = (3.0285*dt*v*(np.tan(delta)^2 + 1))/np.sqrt((0.2500*np.tan(delta)^2 + 1)) - (0.7571*dt*v*np.tan(delta)*np.tan(delta)*(np.tan(delta)*np.tan(delta) + 1))/(0.2500*np.tan(delta)^2 + 1)**1.5

        self.F[3,3] = 1
        self.F[3,5] = dt

        self.F[4,4] = 1
        self.F[4,6] = dt

        self.read_controls()
        if (a_old == 0 or deldot_old == 0):
            self.F[5,5] = 1
            self.F[6,6] = 1
        else:
            self.F[5,5] = self.a/a_old
            self.F[6,6] = self.deldot/deldot_old

    def state_transition_function(self):

        self.x_prior[0] = self.x[0] + self.dt*(self.x[3]*np.cos(self.x[2] + np.arctan(0.5*np.tan(self.x[4]))))
        self.x_prior[1] = self.x[1] + self.dt*(self.x[3]*np.sin(self.x[2] + np.arctan(0.5*np.tan(self.x[4]))))
        self.x_prior[2] = self.x[2] + self.dt*(self.x[3]/0.3302)*np.cos(np.arctan(0.5*np.tan(self.x[4])))*np.tan(self.x[4])
        self.x_prior[3] = self.x[3] + self.dt*self.x[5]
        self.x_prior[4] = self.x[4] + self.dt*self.x[6]
        self.x_prior[5] = self.a
        self.x_prior[6] = self.deldot
        
        
        if self.x_prior[4] >= 0.3:
            self.x_prior[4] = 0.3
        if self.x_prior[4] <= -0.3:
            self.x_prior[4] = -0.3
        if self.x_prior[3] <= 0.2 and self.a != 0.0:
            self.x_prior[3] = 0.2
        if self.x_prior[3] >= 2.0 and self.a != 0.0:
            self.x_prior[3] = 2.0
        if self.x_prior[2] >= 2*3.14159:
            self.x_prior[2] = self.x_prior[2] - 2*3.14159
        if self.x_prior[2] <= -2*3.14159:
            self.x_prior[2] = self.x_prior[2] + 2*3.14159

    def compute_H(self):
        psi = self.x_prior[2]
        a = self.x_prior[5]
        v = self.x_prior[3]
        delta = self.x_prior[4]

        self.H[0, 2] = -a*np.sin(psi)
        self.H[0, 5] = np.cos(psi)
        
        self.H[1,2] = a*np.cos(psi)
        self.H[1,5] = np.sin(psi)

        self.H[2,3] = (3.0285*np.tan(delta))/(0.2500*np.tan(delta)**2 + 1)**0.5
        self.H[2,4] = (3.0285*v*(np.tan(delta)**2 + 1))/(0.2500*np.tan(delta)**2 + 1)**0.5000 - (0.7571*v*np.tan(delta)**2*(np.tan(delta)**2 + 1))/(0.2500*np.tan(delta)**2 + 1)**1.5000

    def compute_residual(self): # Computes difference between measurements and state. Because measurements are acceleration, computes difference between measurements and control inputs
        
        self.h[0] = self.x_prior[5]*np.cos(self.x_prior[2])
        self.h[1] = self.x_prior[5]*np.sin(self.x_prior[2])
        self.h[2] = (self.x_prior[3]/0.3302)*np.cos(np.arctan(0.5*np.tan(self.x_prior[4])))*np.tan(self.x_prior[4])

        self.y = self.z - self.h
        #print(self.h)

    def compute_kalman_gain(self):
        Pprior_Ht = np.matmul(self.P_prior, np.transpose(self.H))
        H_Pprior_Ht_plus_R = np.matmul(self.H, np.matmul(self.P_prior, np.transpose(self.H))) + self.R
        inverse_term = np.linalg.inv(H_Pprior_Ht_plus_R)
        self.K = np.matmul(Pprior_Ht, inverse_term)

    def compute_state(self):
        self.x = self.x_prior + np.matmul(self.K, self.y)
        if self.x[2] >= 2*3.14159:
             self.x[2] = self.x[2] - 2*3.14159
        if self.x[2] <= -2*3.14159:
            self.x[2] = self.x[2] + 2*3.14159
        if self.x_[4] >= 0.3:
            self.x_[4] = 0.3
        if self.x[4] <= -0.3:
            self.x[4] = -0.3
        if self.x[3] <= 0.2 and self.a != 0.0:
            self.x[3] = 0.2
        if self.x[3] >= 2.0 and self.a != 0.0:
            self.x[3] = 2.0
        
        
    
    def compute_covariance(self):
        I_minus_KH = np.eye(7) - np.matmul(self.K, self.H)
        I_minus_KH_t = np.transpose(I_minus_KH)
        K_R_Kt = np.matmul(self.K, np.matmul(self.R, np.transpose(self.K)))
        self.P_next = np.matmul(I_minus_KH, np.matmul(self.P_prior, I_minus_KH_t)) + K_R_Kt

    def update_step(self):
        self.compute_H()
        #print('H matrix: \n')
        #print(self.H)
        #print('\n')
        
        self.read_IMU()
        print('Measurement z: \n')
        print(self.z)
        print('\n')        
        
        self.compute_residual()
        #print('Measurement function h: \n')
        #print(self.h)
        #print('\n')             
        
        #print('Residual y = (z -h): \n')
        #print(self.y)
        #print('\n')        
        
        
        self.compute_kalman_gain()
        #print('Kalman gain K: \n')
        #print(self.K)
        #print('\n')        
        
        self.compute_state()
        #print('State update x: \n')
        #print(self.x)
        #print('\n')
        
        self.compute_covariance()
        #print('P_next: \n')
        #print(self.P_next)
        #print('\n')        

    def acknowledge_time_passed(self):
        self.x = self.x
        self.P_prev = self.P_next


def main(args=None):
    rclpy.init(args=args)
    car_kf = F1TenthKalmanFilter()

    rclpy.spin(car_kf)

    car_kf.destroy_node()
    car_kf.zed.close()


if __name__ == "__main__":
    main()
