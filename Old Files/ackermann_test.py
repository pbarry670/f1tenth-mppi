#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class driveTest(Node):
    def __init__(self):
        super().__init__("ack_test") #Name of node
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE, history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, depth=1)
        self.steer = 0.0
        self.accel = 0.0
        self.ack_publisher = self.create_publisher(AckermannDriveStamped, "/drive", qos_profile=qos_profile)
        self.timer = self.create_timer(0.02, self.publish_ackermann)

    def publish_ackermann(self):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rclpy.time.Time().to_msg()
        ack_msg.header.frame_id = "laser"
        ack_msg.drive.steering_angle= self.steer
        ack_msg.drive.acceleration = self.accel
        self.ack_publisher.publish(ack_msg)

    def set_speed(self, speed_cmd):
        self.speed = speed_cmd

    def set_steer(self, steer_cmd):
        self.steer = steer_cmd

    def set_accel(self, accel_cmd):
        self.accel = accel_cmd


def main(args=None):
    rclpy.init()
    driver = driveTest()

    flag = True
    while (flag):
        accel_cmd = 0.1
        steer_cmd = 0.1
        driver.set_accel(accel_cmd)
        driver.set_steer(steer_cmd)
        rclpy.spin_once(driver)
        time.sleep(1.0)

        accel_cmd = 0.05
        steer_cmd = -0.1
        driver.set_accel(accel_cmd)
        driver.set_steer(accel_cmd)
        rclpy.spin_once(driver)
        time.sleep(1.0)


    rclpy.shutdown(driver)


if __name__ == '__main__':
    main()


