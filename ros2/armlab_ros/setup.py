from glob import glob

from setuptools import setup

package_name = "armlab_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bogdan-Marian Dumitru",
    maintainer_email="dbogdan3002@gmail.com",
    description="ROS 2 visualisation for the armlab kinematics and planning "
                "library",
    entry_points={
        "console_scripts": [
            "plan_node = armlab_ros.plan_node:main",
        ],
    },
)
