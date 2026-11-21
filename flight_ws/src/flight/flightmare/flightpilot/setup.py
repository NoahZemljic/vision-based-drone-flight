from setuptools import setup
import os
from glob import glob

package_name = 'flightpilot'

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
    description='ROS 2 deployment nodes for the VBDF course tracking/hover policies (port of the ROS 1 flightpilot)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'run_hover_pilot_isaac = flightpilot.run_hover_pilot_isaac:main',
            'run_tracking_pilot_isaac = flightpilot.run_tracking_pilot_isaac:main',
            'virtual_actor_detector = flightpilot.virtual_actor_detector:main',
        ],
    },
)
