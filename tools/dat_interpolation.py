"""
This program reads from the telemetry of the UR5
"""
import sys
import numpy as np


def sample_interpolate(data, sample_freq=10):
    """
    Samples from every 10 time stamps, and linearly interpolates between them

    returns an array
    """
    out = [data[0]]
    for i in range(len(data)):
        if i % sample_freq == 0:
            tmp = np.linspace(out[-1], data[i], sample_freq)
            out = np.vstack([out, tmp])
    return out


path_to_file = sys.argv[1:][0]
print("Reading dat file from :", path_to_file)

contents = []
with open(path_to_file) as f:
    contents = f.readlines()

waypoints = np.array([0] * 6)

for string in contents:
    arm_joint = str.split(string)[1:7]
    for i in range(len(arm_joint)):
        arm_joint[i] = (float)(arm_joint[i])
    waypoints = np.vstack([waypoints, arm_joint])

waypoints = waypoints[1:]
print(sample_interpolate(waypoints))
