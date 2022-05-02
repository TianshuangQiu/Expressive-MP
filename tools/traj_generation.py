from cProfile import label
import math
from textwrap import wrap
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.optimize


TIMESTEP = 0.001  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 25  # FPS in original video, should be 25
prev_hand_kp = [0, 0]
parser = argparse.ArgumentParser(
    description='Generate a trajectory from parsed JSON files')
parser.add_argument('file_path', type=str,
                    help='Where is the numpy file stored at?')
parser.add_argument('out_path', type=str,
                    help='Where should I save the output file?')
parser.add_argument('filter', type=str,
                    help='Current options: fft, cc')

# '/home/akita/autolab/ur5/ur5_description/urdf/ur5_joint_limited_robot.urdf'
args = parser.parse_args()
with open(args.file_path, "rb") as f:
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


def sinfunc(t, A, w, p, c):
    return A * np.sin(w*t + p) + c


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    # excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    try:
        popt, pcov = scipy.optimize.curve_fit(
            sinfunc, tt, yy, p0=guess, maxfev=10000)
    except:
        return {"fitfunc": lambda x: 0}
    A, w, p, c = popt
    f = w/(2.*np.pi)
    def fitfunc(t): return A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc}


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
    horizontal = np.split(sh, sh_zeros) + \
        np.split(el, el_zeros) + \
        np.split(wr, wr_zeros)
    split = np.array(horizontal, dtype=object)

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
        return np.max(np.correlate(arr1, arr2, mode='full')) * np.min([norm1, norm2])

    cc = np.zeros((split.shape[0], split.shape[0]))
    for i in range(len(split)):
        for j in range(len(split)):
            val = cross(split[i], split[j])
            cc[i][j] = 0 if i == j else val
    cc = np.max(cc, axis=0)
    sort = np.argsort(cc)

    # for i in range(1, 5):
    #     curr_dat = horizontal[sort[i]]
    #     x_dat = np.arange(len(curr_dat))
    #     plt.plot(curr_dat, label=f"{i}th principle component")
    #     rst = fit_sin(x_dat, curr_dat)["fitfunc"]
    #     plt.plot(np.array(list(map(rst, x_dat))), label=f"{i}th fit")
    # # plt.scatter(sh_zeros, sh[sh_zeros], c='r', label="split points")
    # plt.title("Extracted Vectors")
    # plt.xlabel("Time (steps)")
    # plt.ylabel("Angle (rad)")
    # ax = plt.legend()
    # plt.show()

    principle_motion = np.array(t)
    for i in range(1, len(sort)//2):
        curr_dat = horizontal[sort[i]]
        x_dat = np.arange(len(curr_dat))
        rst = fit_sin(x_dat, horizontal[sort[i]])["fitfunc"]
        principle_motion = np.vstack([principle_motion, list(map(rst, t))])

    principle_motion = principle_motion[1:]
    print(f"Computed {len(principle_motion)} principle motions")

    def get_rand_motion(k):
        rand_idx = np.random.randint(len(principle_motion), size=k)
        out = np.zeros(len(principle_motion[0]))
        for i in rand_idx:
            out = out + principle_motion[i]

        out /= (np.max(out)-np.min(out))
        out *= 2 * np.pi
        return out

    position = np.vstack([get_rand_motion(20),
                          sh,
                          el,
                          fourier_filter(wr, 30),
                          get_rand_motion(5),
                          get_rand_motion(5)/5,
                          ])

    for p in range(len(position)):
        plt.plot(position[p], label=f"joint {p}")
    ax = plt.legend()
    plt.show()

    position = position.T
    position = linear_interp(position, int(1 / FRAMERATE / TIMESTEP))
    t_int = linear_interp(t[np.newaxis].T, int(1 / FRAMERATE / TIMESTEP))

    velocity = num_deriv(position, TIMESTEP)
    acceleration = num_deriv(velocity, TIMESTEP)
    jerk = num_deriv(acceleration, TIMESTEP)

    output = np.hstack([t_int, position, velocity, acceleration, jerk])
    output = np.round_(output, decimals=5)
    np.savetxt(args.out_path, output, fmt="%10.5f", delimiter='\t')


def do_decomp():
    sh_d = np.fft.fft(sh)/2116
    plt.plot(sh, label="original")
    total = np.zeros(2116)
    for i in range(100):
        c = sh_d[i]
        print(c)
        def f(x): return np.exp(x*c).real
        total = list(map(f, t))
    plt.plot(total, label="summed")
    plt.legend()
    plt.show()


# Rest of program
if (args.filter == "fft"):
    do_fft()
elif (args.filter == "cc"):
    do_cc()
elif (args.filter == "decomp"):
    do_decomp()
else:
    print("Option not implemented yet")
