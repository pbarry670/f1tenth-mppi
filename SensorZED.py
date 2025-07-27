########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

## 

class SensorZED(Node):
    def __init__(self):
        super().__init__("zed_sensor_gen") # Name of node
        self.sensor_data_publisher = self.create_publisher(Float32MultiArray, "/zed_sensor_data", 10)
        
        self.t_imu = sl.Timestamp()
        self.t_baro = sl.Timestamp()
        self.t_mag = sl.Timestamp()
        
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.update_data)
        
    def publish_sensor_data(self, a, w):
        sensor_data = [a[0], a[1], a[2], w[0], w[1], w[2]]
        self.sensor_data_to_publish = Float32MultiArray(data=sensor_data)
        self.sensor_data_publisher.publish(self.sensor_data_to_publish)
        
        
    def update_data(self):
        sensors_data = sl.SensorsData()
        if self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS :
            # Check if the data has been updated since the last time
            # IMU is the sensor with the highest rate
            if self.is_new(sensors_data.get_imu_data()):
                quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
                angular_velocity = sensors_data.get_imu_data().get_angular_velocity()
                if self.is_new(sensors_data.get_magnetometer_data()):
                    magnetic_field_calibrated = sensors_data.get_magnetometer_data().get_magnetic_field_calibrated()
                self.publish_sensor_data(linear_acceleration, angular_velocity)
    
    ##
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.MagnetometerData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_mag.get_microseconds())
            if new_:
                self.t_mag = sensor.timestamp
            return new_

##
#  Function to display sensor parameters
def printSensorParameters(sensor_parameters):
    if sensor_parameters.is_available:
        print("*****************************")
        print("Sensor type: " + str(sensor_parameters.sensor_type))
        print("Max rate: "  + str(sensor_parameters.sampling_rate) + " "  + str(sl.SENSORS_UNIT.HERTZ))
        print("Range: "  + str(sensor_parameters.sensor_range) + " "  + str(sensor_parameters.sensor_unit))
        print("Resolution: " + str(sensor_parameters.resolution) + " "  + str(sensor_parameters.sensor_unit))
        if not math.isnan(sensor_parameters.noise_density):
            print("Noise Density: "  + str(sensor_parameters.noise_density) + " " + str(sensor_parameters.sensor_unit) + "/√Hz")
        if not math.isnan(sensor_parameters.random_walk):
            print("Random Walk: "  + str(sensor_parameters.random_walk) + " " + str(sensor_parameters.sensor_unit) + "/s/√Hz")
    


def main():
    # Create a Camera object
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Get camera information sensors_data
    info = zed.get_camera_information()

    cam_model = info.camera_model
    if cam_model == sl.MODEL.ZED :
        print("This tutorial only supports ZED-M and ZED2 camera models, ZED does not have additional sensors")
        exit(1)

    # Display camera information (model,S/N, fw version)
    print("Camera Model: " + str(cam_model))
    print("Serial Number: " + str(info.serial_number))
    print("Camera Firmware: " + str(info.camera_configuration.firmware_version))
    print("Sensors Firmware: " + str(info.sensors_configuration.firmware_version))

    # Display sensors parameters (imu,barometer,magnetometer)
    printSensorParameters(info.sensors_configuration.accelerometer_parameters) # accelerometer configuration
    printSensorParameters(info.sensors_configuration.gyroscope_parameters) # gyroscope configuration
    printSensorParameters(info.sensors_configuration.magnetometer_parameters) # magnetometer configuration
    printSensorParameters(info.sensors_configuration.barometer_parameters) # barometer configuration
    
    # Used to store the sensors timestamp to know if the sensors_data is a new one or not
    rclpy.init()
    ts_handler = SensorZED()
    ts_handler.zed = zed
    rclpy.spin(ts_handler)
    
    ts_handler.destroy_node() 
    zed.close()
    return 0

if __name__ == "__main__":
    main()

