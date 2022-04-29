import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos

"""
Pass the path of the .dat file to the program as a command line argument.
The output gif will be in /trajs
"""

path_to_file = sys.argv[1:][0]
contents = []
with open(path_to_file) as f:
    contents = f.readlines()

arr = np.array([0] * 3)
print("Read lines: ", len(contents))
for string in contents:
    arm_joint = str.split(string)[1:7]
    vert = np.array(
        [(float)(arm_joint[1]), (float)(arm_joint[2]), (float)(arm_joint[3])]
    )
    arr = np.vstack([arr, vert])

arr = arr[1:]
fig = plt.figure()
ax = plt.axes(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
(line,) = ax.plot([], [], lw=3)


def animate(i):
    angles = arr[i]

    # Unrolled loop for .1% performance increase kekW
    x = [0]
    y = [0]
    x_val = 2 * cos(angles[0])
    y_val = 2 * sin(-angles[0])
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    x_val = 2 * cos(angles[1] + angles[0])
    y_val = 2 * (sin(-angles[1] - angles[0]))
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    x_val = 0.5 * cos(angles[2] + angles[1] + angles[0])
    y_val = 0.5 * sin(-angles[2] - angles[1] - angles[0])
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    line.set_data(x, y)

    return (line,)


anim = FuncAnimation(
    fig,
    animate,
    frames=range(0, len(arr), (int)(len(arr) / 500)),
    interval=20,
    blit=True,
)
anim.save("trajs" + os.sep + path_to_file.split(os.sep)[-1][:-4] + ".gif")
