from cProfile import label
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.misc import derivative

TIMESTEP = 0.001  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 25  # FPS in original video, should be 25
prev_hand_kp = [0, 0]
with open("saves/IMG_4015", "rb") as f:
    stack = np.load(f)

sh = stack[0]
el = stack[1]
wr = stack[2]
t = np.arange(len(sh)) / FRAMERATE


def linear_interp(array, num):
    interpolated = array[0]
    for i in range(len(array) - 1):
        interp = np.linspace(array[i], array[i + 1], num)
        interpolated = np.vstack([interpolated, interp])

    return interpolated[1:]


def fourier_filter(array, thresh):
    fourier = np.fft.rfft(array)
    fourier[np.abs(fourier) < thresh] = 0
    filtered = np.fft.irfft(fourier)
    return filtered


def num_deriv(array, t):
    stack = None
    for a in array.T:
        grad = np.gradient(a, t, axis=0)
        stack = (grad if stack is None else np.vstack([stack, grad]))
    return stack.T


def do_fft():
    sh_filtered = fourier_filter(sh, 25)
    el_filtered = fourier_filter(el, 15)
    wr_filtered = fourier_filter(wr, 10)

    position = np.vstack([np.zeros_like(t),
                          # np.sin(np.arange(len(data[0])) * 2 * np.pi / len(data[0])) * 0.5,
                          sh_filtered,
                          el_filtered,
                          wr_filtered,
                          np.ones_like(t) * np.pi / 2,
                          np.zeros_like(t)
                          ]).T

    position = linear_interp(position, int(1 / FRAMERATE / TIMESTEP))
    t_int = linear_interp(t[np.newaxis].T, int(1 / FRAMERATE / TIMESTEP))

    velocity = num_deriv(position, TIMESTEP)
    acceleration = num_deriv(velocity, TIMESTEP)
    jerk = num_deriv(acceleration, TIMESTEP)

    output = np.hstack([t_int, position, velocity, acceleration, jerk])
    output = np.round_(output, decimals=5)
    np.savetxt("IMG_4015.dat", output, fmt="%10.5f", delimiter='\t')


def do_cc():

    sh_d = np.abs(np.gradient(sh, axis=0))
    el_d = np.abs(np.gradient(el, axis=0))
    wr_d = np.abs(np.gradient(wr, axis=0))

    def filter_zeros(arr):
        """
        Arr should be non-negative
        """
        zeros = np.where(arr < 0.01)
        zeros = zeros[0]
        prev = zeros[0]
        out = np.array([prev])
        flip = True
        for i in zeros:
            if (i > prev+20):
                idx = (i+out[-1])//2
                flucuation = np.abs(arr[idx]-arr[i])
                if (flucuation > 0.005):
                    if flip:
                        out = np.hstack([out, i])
                    flip = not flip
                prev = i
        return out

    sh_zeros = filter_zeros(sh_d)
    el_zeros = filter_zeros(el_d)
    wr_zeros = filter_zeros(wr_d)
    split = np.array(np.split(sh, sh_zeros) +
                     np.split(el, el_zeros) +
                     np.split(wr, wr_zeros),
                     dtype=object)

    def cross(arr1, arr2):
        """
        Cross correlation in case the two vectors do not 
        have the same length
        """
        length = np.max([len(arr1), len(arr2)])
        norm1, norm2 = np.linalg.norm(arr1), np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0
        arr1 = arr1 / norm1
        arr2 = arr2 / norm2
        arr1 = np.pad(arr1, length)
        arr2 = np.pad(arr2, length)
        return np.max(np.correlate(arr1, arr2, mode='full'))

    cc = np.zeros((split.shape[0], split.shape[0]))
    for i in range(len(split)):
        for j in range(len(split)):
            val = cross(split[i], split[j])
            cc[i][j] = 0 if val == 1 else val

    cc = np.max(cc, axis=1)
    print(cc)
    fig, ax = plt.subplots()
    for s in split:
        plt.plot(s)
    # plt.scatter(sh_zeros, sh[sh_zeros], c='r', label="split points")
    plt.title("Extracted Vectors")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (rad)")
    ax = plt.legend()
    plt.show()


# Rest of program
parser = argparse.ArgumentParser(
    description='Generate a trajectory from parsed JSON files')
parser.add_argument('filter', type=str,
                    help='Current options: fft, cc')
# '/home/akita/autolab/ur5/ur5_description/urdf/ur5_joint_limited_robot.urdf'
args = parser.parse_args()

if (args.filter == "fft"):
    do_fft()
elif (args.filter == "cc"):
    do_cc()
else:
    print("Option not implemented yet")
