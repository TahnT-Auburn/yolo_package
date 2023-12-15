from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    config_path = "/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/yolo_package/config/yolo_config.yaml"
    
    return LaunchDescription([
        Node(
            package="yolo_package",
            executable="yolo_object_detection",
            name="object_detection",
            output="screen",
            emulate_tty=True,
            parameters=[config_path]
        ),
    ])