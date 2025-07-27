#!usr/bin/env python3

# Envisions the state estimation prediction provided by the Kalman filter

import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

class KalmanVisualizer(Node):
    def __init__(self):
        super().__init__("kf_viz")
        self.state_receiver = self.create_subscription(Float32MultiArray, "/filtered_state", self.state_read_callback, 10)
        self.covariance_receiver = self.create_subscription(Float32MultiArray, "/covariances", self.covariance_read_callback, 10)

        self.max_queue_size = 200
        
        self.x_history = np.zeros(self.max_queue_size)
        self.y_history = np.zeros(self.max_queue_size)
        self.psi_history = np.zeros(self.max_queue_size)
        self.xcov_history = np.zeros(self.max_queue_size)
        self.ycov_history = np.zeros(self.max_queue_size)
        self.psicov_history = np.zeros(self.max_queue_size)
        
        self.reported_state = np.zeros(8)
        self.reported_covs = np.zeros(3)
        
        self.update_rate = 0.1 # 10 Hz

        self.oldest_data_time = -self.update_rate*self.max_queue_size
        self.time_axis = np.linspace(self.oldest_data_time, 0, self.max_queue_size) # Horizontal time axis in terms of "seconds ago"

        self.is_first_time = True
        self.timer = self.create_timer(self.update_rate, self.update_visualization) # Run update_visualization() every update_rate seconds

    def update_visualization(self):

        if self.is_first_time:
            plt.ion()
            self.fig, (self.locplot, self.headingplot, self.covplot) = plt.subplots(3)
            
            self.line1, = self.locplot.plot(self.x_history, self.y_history)
        
            self.line2, = self.headingplot.plot(self.time_axis, self.psi_history)
		
            self.line3, = self.covplot.plot(self.time_axis, self.xcov_history, "-b", label="xcov")
            self.line4, = self.covplot.plot(self.time_axis, self.ycov_history, "-r", label="ycov")
            self.line5, = self.covplot.plot(self.time_axis, self.psicov_history, "-g", label="psicov")
            
            self.locplot.set_title('Filtered (X, Y) Location of Car')
            self.headingplot.set_title('Filtered Steering Angle History')
            self.covplot.set_title('Covariance History of X, Y, and Steer Angle')
        
            self.locplot.set_xlabel('X Position (m)')
            self.locplot.set_ylabel('Y Position (m)')
            self.headingplot.set_xlabel('Number of Seconds Ago')
            self.headingplot.set_ylabel('Steer Angle (rad)')
            self.covplot.set_xlabel('Number of Seconds Ago')
            self.covplot.set_ylabel('Covariance Values')
        else:
            self.fig.canvas.flush_events()
            self.line1.set_xdata(self.x_history)
            self.line1.set_ydata(self.y_history)
            self.line2.set_ydata(self.psi_history)
            self.line3.set_ydata(self.xcov_history)
            self.line4.set_ydata(self.ycov_history)
            self.line5.set_ydata(self.psicov_history)
            
            self.locplot.set_xlim([np.min(self.x_history) - 1, np.max(self.x_history) + 1])
            self.locplot.set_ylim([np.min(self.y_history) - 1, np.max(self.y_history) + 1])
            
            xcov_max = np.max(self.xcov_history)
            ycov_max = np.max(self.ycov_history)
            psicov_max = np.max(self.psicov_history)
            maxcov_arr = np.array([xcov_max, ycov_max, psicov_max])
            
            self.headingplot.set_ylim([np.min(self.psi_history) - 0.5, np.max(self.psi_history) + 0.5])
            self.covplot.set_ylim([0, np.max(maxcov_arr) + 1])
            
            self.fig.canvas.draw()
            
            
        self.is_first_time = False

        #plt.show(block=False)
        #plt.pause(0.01)
        self.update_data()


    def state_read_callback(self, msg : Float32MultiArray):
        self.reported_state = msg.data #[x, y, psi, xdot, ydot, psidot, xdotdot, ydotdot]
        #print(self.reported_state)

    def covariance_read_callback(self, msg : Float32MultiArray):
        self.reported_covs = msg.data #[x_covariance, y_covariance, psi_covariance]
        #print(self.reported_covs)

    def update_data(self): # Keep the reported state/covariance history at the length of the max queue size

        self.x_history = np.append(self.x_history, self.reported_state[0])
        self.y_history = np.append(self.y_history, self.reported_state[1])
        self.psi_history = np.append(self.psi_history, self.reported_state[2])
        self.xcov_history = np.append(self.xcov_history, self.reported_covs[0])
        self.ycov_history = np.append(self.ycov_history, self.reported_covs[1])
        self.psicov_history = np.append(self.psicov_history, self.reported_covs[2])
        
        self.x_history = np.delete(self.x_history, 0)
        self.y_history = np.delete(self.y_history, 0)
        self.psi_history = np.delete(self.psi_history, 0)
        self.xcov_history = np.delete(self.xcov_history, 0)
        self.ycov_history = np.delete(self.ycov_history, 0)
        self.psicov_history = np.delete(self.psicov_history, 0)

def main(args=None):
    rclpy.init(args=args)
    kf_viz = KalmanVisualizer()
    rclpy.spin(kf_viz)
    kf_viz.destroy_node()

if __name__ == "__main__":
    main()
