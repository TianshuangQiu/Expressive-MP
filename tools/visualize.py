from cmath import pi
from urdfpy import URDF
from urdfpy import SafetyController
import numpy as np
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Visualize a UR5 Trajectory')
parser.add_argument('file_dir', type=str,
                    help='Location for the trajectory file')
# '/home/akita/autolab/ur5/ur5_description/urdf/ur5_joint_limited_robot.urdf'
args = parser.parse_args()


# Load urdf file
robot = URDF.load("/home/akita/autolab/urdfpy/tests/data/ur5/ur5.urdf")

for joint in robot.actuated_joints:
    joint.safety_controller = SafetyController(0)

# Load traj file
contents = []
with open(args.file_dir) as f:
    contents = f.readlines()

arr = np.array([0] * 6)
print("Read lines: ", len(contents))
curr_time = 0
for string in contents:
    arm_joint = str.split(string)[0:7]
    if ((float)(arm_joint[0]) > curr_time):
        vert = np.array(
            [(float)(arm_joint[1]), (float)(arm_joint[2]), (float)(arm_joint[3]),
             (float)(arm_joint[4]), (float)(arm_joint[5]), (float)(arm_joint[6])]
        )
        curr_time += 0.016
        arr = np.vstack([arr, vert])

arr = arr.T

real_time = (float)(str.split(contents[-1])
                    [0]) - (float)(str.split(contents[0])[0])

print(f"Real time calculated to be: {real_time}")
robot.animate(cfg_trajectory={robot.actuated_joints[0]: arr[0],
                              robot.actuated_joints[1]: arr[1],
                              robot.actuated_joints[2]: arr[2],
                              robot.actuated_joints[3]: arr[3],
                              robot.actuated_joints[4]: arr[4],
                              robot.actuated_joints[5]: arr[5]},
              loop_time=real_time)
