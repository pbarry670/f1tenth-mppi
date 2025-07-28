# f1tenth-mppi
Code written for the testing and implementation of a Model-Predictive Path Integral (MPPI) controller on Georgia Tech's Dynamics & Control System Lab's F1/10th autonomous racecar fleet. MPPI is a type of nonlinear control algorithm growing in popularity in the field of robotics; it leverages the computational power of GPU processing to solve for a lowest-cost trajectory over a specified time horizon.
- Includes speed-testing code for various libraries and raw CUDA. PyTorch was selected for GPU acceleration architecture.
- Includes MPPIControl.py, which is wrapped in a ROS node and contains the MPPI algorithm.
- Includes CenterlineVisualization.py, which acts as a ROS node, taking in LiDAR information from the car and transforming it into a centerline.
- Includes ackermann_test.py, which shows the way in which AckermannDriveStamped messages may be published to the car using control of just the acceleration and steering angle.

# Using MPPIControl.py

First, to run MPPIControl.py, the code that runs an MPPI control algorithm for an f1tenth car, one must have ROS2 Eloquent, PyTorch, and rclpy installed. The program acts as a ROS node that subscribes to /scan LaserScan messages from a LiDAR and publishes AckermannDriveStamped messages to a /drive topic. 
This program should be placed in a ROS2 package from which it can be run from a Linux terminal. To create a ROS2 package, navigate to the /src folder of your ROS2 workspace and create a package called "mppi_controller" that works with Python using the command

```
ros2 pkg create mppi_controller --build-type ament_python --dependecies rclpy
```
Navigate to mppi_controller > mppi_controller and place MPPIControl.py here. 
Now, open up setup.py, and within the brackets of "console_scripts" write the line 
```
"mppi = mppi_controller.MPPIControl:main"
```
This allows the node to be called using ROS functionality. After using the command "colcon_build --symlink-install" start the car using the command 
```
ros2 launch f1tenth_stack bringup_launch.py
```
And begin run the "mppi" node with the command
```
ros2 run mppi_controller mppi
```
The "mppi" node will subscribe to the /scan topic to process LaserScan messages and will published AckermannDriveStamped messages to /drive, commanding steering angle and acceleration.

# Using CarlessMPPI.py
The same instructions apply above for to get CarlessMPPI.py to run, but since this program may be run without a car and is purely for testing purposes there are no other nodes. As a result, after entering the node definition into setup.py within the mppi_controller package, modifying the node name to be "carless_mppi"...
```
"carless_mppi = mppi_controller.MPPIControl:main"
```
...the node should be able to run without running the f1tenth_stack bringup_launch.py file and may simply begin running using 
```
ros2 run mppi_controller carless_mppi
```
assuming the workspace has been properly built after adding the new package and changes to its setup.py file.

# The MPPI Algorithm

The following is an outline of how the MPPI algorithm works:
1. A sequence of acceleration and steering commands are generated with a normal distribution. These are stored in arrays with as many rows as the number of time steps in the time horizon and as many columns as there are trajectories.
2. State vectors are generated; there are as many state vectors as there are trajectories, and they are stored as columns in an array. Each state vector is [x_position, y_position, heading_angle, longitudinal_velocity, laterial_velocity]. x is in the longitudinal direction, and y is in the lateral direction.
3. A cost array is initialized as an array of zeroes with a length equal to the number of trajectories.
4. Using LiDAR measurements, a track centerline is generated as a series of discrete points. A cubic spline is fit to these discrete points.
5. Over the time horizon, the dynamics are forwarded and the cost is computed.
   - Dynamics are forwarded according to the kinematic bicycle model using the random control inputs for each trajectory.
   - Cost is computed with position cost (uses position from the track centerline, found using centerline spline), heading cost (another spline that maps x position to an ideal heading angle and finds the difference with the car's heading angle according to the dynamics), velocity cost (using the difference between the car's velocity using its dynamics and a set desired velocity) and control input cost (which penalizes control inputs of large magnitude). 
6. After all costs have been computed and cumulatively added over the time horizon, the trajectory with the lowest cost is found.
7. The first control inputs in the control sequence associated with the lowest-cost trajectory are taken.
8. These control inputs are used to actuate the car.
9. The cost array is reset to zero, and so is the x position of the car, as the x=0 of the reference frame we are using moves with the car.
10. New control inputs are generated with the mean set as the control inputs that were found in step 7.
11. Repeat steps 4 - 10 in a loop.
