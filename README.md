# yolo_package

## Description:
ROS2 package to run YOLOv3 object detection.

This package runs YOLOv3 through OpenCV's Deep Neural Networks (DNN) module and outputs YOLO detections on an input image or video.

**For more information on this package, check the Joplin documentation under the _Utilities Package_ page.**

## Requirements:
* Ubuntu Focal Fossa
* ROS2 Foxy Fitzroy
* C++17 or higher

## To Use:
**_Before Use:_**
* **Make sure ALL PATHS ARE SET CORRECTLY in the launch and config files before use!**
* **These steps assume you have already created a workspace folder and a `/src` directory within it!**

**_Steps:_**
1. Navigate into the `/src` directory of your workspace and clone the repo using `git clone`
2. Navigate back into the workspace directory and source `$ source /opt/ros/foxy/setup.bash`
3. Build package `$ colcon build` or `$ colcon build --packages-select <package_name>`
4. Open a new terminal and source it `$ . install/setup.bash`
5. Run launch file `$ ros2 launch <package_name> <launch_file_name>` in this case it is `$ ros2 launch yolo_pacakge yolo_launch.py`