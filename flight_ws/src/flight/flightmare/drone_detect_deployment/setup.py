import os
from glob import glob

from setuptools import setup

package_name = 'drone_detect_deployment'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yunlong Song',
    maintainer_email='song@ifi.uzh.ch',
    description='ROS 2 YOLOv5 detector node for the VBDF course',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = drone_detect_deployment.yolo_detector:main',
        ],
    },
)
