import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos

path_to_file = sys.argv[1:][0]
contents = []
with open(path_to_file) as f:
    contents = f.readlines()

arr = np.array([0] * 3)
print("Read lines: ", len(contents))
for string in contents:
    arm_joint = str.split(string)[1:7]
    vert = np.array(
        [(float)(arm_joint[1]), (float)(arm_joint[2]), (float)(arm_joint[-2])]
    )
    arr = np.vstack([arr, vert])

arr = arr[1:]
fig = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
(line,) = ax.plot([], [], lw=3)


def animate(i):
    angles = arr[i]
    x = [0]
    y = [0]
    x_val = cos(angles[0])
    y_val = sin(angles[0])
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    x_val = cos(angles[1] + angles[0])
    y_val = sin(angles[1] + angles[0])
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    x_val = cos(angles[2] + angles[1] + angles[0])
    y_val = sin(angles[2] + angles[1] + angles[0])
    x_prev = x[-1]
    y_prev = y[-1]
    x_tmp = np.linspace(x_prev, x_prev + x_val, 100)
    y_tmp = np.linspace(y_prev, y_prev + y_val, 100)
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])
    line.set_data(x, y)

    return (line,)


anim = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
anim.save("vertical_joints.gif")
