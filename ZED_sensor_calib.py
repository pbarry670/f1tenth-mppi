import cv2
import numpy as np
import math
import time 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class SensorCalib(Node):
    def __init__(self):
    
       super().__init__("sensor_calib")
       self.receive_sensors = self.create_subscription(Float32MultiArray, "/zed_sensor_data", self.sensor_read_callback, 10)
       
       self.num_samples_taken = 1
       
       self.accel_x_mean = 0.0
       self.newest_accel_x = 0.0
       self.accel_x_measurements = np.array([0.0])
       
       self.accel_y_mean = 0.0
       self.newest_accel_y = 0.0
       self.accel_y_measurements = np.array([0.0])
       
       self.gyro_z_mean = 0.0
       self.newest_gyro_z = 0.0
       self.gyro_z_measurements = np.array([0.0])
       
       self.dt = 0.01
       self.timer = self.create_timer(self.dt, self.update_variance)       
   
    def sensor_read_callback(self, msg : Float32MultiArray):
        z = msg.data
        self.newest_accel_x = z[2]
        self.newest_accel_y = z[0]
        self.newest_gyro_z = z[4]
        
    def update_variance(self):
        if (self.num_samples_taken < 10000):
            self.accel_x_measurements = np.append(self.accel_x_measurements, self.newest_accel_x)
            self.accel_y_measurements = np.append(self.accel_y_measurements, self.newest_accel_y)
            self.gyro_z_measurements = np.append(self.gyro_z_measurements, self.newest_gyro_z)
            self.num_samples_taken = self.num_samples_taken + 1
        
        self.accel_x_mean = np.average(self.accel_x_measurements)
        self.accel_y_mean = np.average(self.accel_y_measurements)
        self.gyro_z_mean = np.average(self.gyro_z_measurements)
   
        self.variance_accel_x = np.sum((self.accel_x_measurements - self.accel_x_mean)*(self.accel_x_measurements - self.accel_x_mean))/(self.num_samples_taken - 1)
        self.variance_accel_y = np.sum((self.accel_y_measurements - self.accel_y_mean)*(self.accel_y_measurements - self.accel_y_mean))/(self.num_samples_taken - 1)
        self.variance_gyro_z = np.sum((self.gyro_z_measurements - self.gyro_z_mean)*(self.gyro_z_measurements - self.gyro_z_mean))/(self.num_samples_taken - 1)
        
        print("\n")
        print("Updated Variance Values: \n")
        print(f"Accel-X Variance: {self.variance_accel_x} \n")
        print(f"Accel-X Mean: {self.accel_x_mean} \n")
        print(f"Accel-Y Variance: {self.variance_accel_y} \n")
        print(f"Accel-Y Mean: {self.accel_y_mean} \n")
        print(f"Gyro-Z Variance: {self.variance_gyro_z} \n")
        print(f"Gyro-Z Mean: {self.gyro_z_mean} \n")
        print(f"Number of Samples Taken: {self.num_samples_taken} \n")

         
         
def main():
    rclpy.init()
    sensor_calib = SensorCalib()
    rclpy.spin(sensor_calib)
    
    sensor_calib.destroy_node()
    
if __name__ == "__main__":
    main()

