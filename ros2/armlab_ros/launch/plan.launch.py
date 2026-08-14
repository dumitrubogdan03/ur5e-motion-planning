from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    obstacles = ParameterValue(LaunchConfiguration("obstacles"),
                               value_type=int)
    seed = ParameterValue(LaunchConfiguration("seed"), value_type=int)
    speed = ParameterValue(LaunchConfiguration("speed"), value_type=float)
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("armlab_ros"), "rviz", "plan.rviz"])

    return LaunchDescription([
        DeclareLaunchArgument("obstacles", default_value="20"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("speed", default_value="0.6"),
        Node(
            package="armlab_ros",
            executable="plan_node",
            name="armlab_plan",
            output="screen",
            parameters=[{"obstacles": obstacles,
                         "seed": seed,
                         "speed": speed}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
        ),
    ])
