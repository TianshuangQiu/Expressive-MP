from os import remove
from tqdm import tqdm
import math
from operator import pos
from textwrap import wrap
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.optimize


TIMESTEP = 0.008  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 30  # FPS in original video, should be 25
prev_hand_kp = [0, 0]
parser = argparse.ArgumentParser(
    description='Generate a trajectory from parsed JSON files')
parser.add_argument('file_path', type=str,
                    help='Where is the numpy file stored at?')
parser.add_argument('out_path', type=str,
                    help='Where should I save the output file?')
parser.add_argument('filter', type=str,
                    help='Current options: fft, cc')

args = parser.parse_args()
with open(args.file_path, "rb") as f:
    stack = np.load(f)

sh = stack[0]
el = stack[1]
wr = stack[2]
rt = stack[3]
t = np.arange(len(sh)) / FRAMERATE


def linear_interp(array, num):
    interpolated = array[0]
    print("Linearly interpolating")
    for i in tqdm(range(len(array) - 1)):
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


def save_traj(position):
    div = int(1/TIMESTEP)
    position = linear_interp(position, div)
    t_int = linear_interp(t[np.newaxis].T, div)

    velocity = num_deriv(position, TIMESTEP/FRAMERATE)
    acceleration = num_deriv(velocity, TIMESTEP/FRAMERATE)
    jerk = num_deriv(acceleration, TIMESTEP/FRAMERATE)

    output = [0]*19

    print("Filtering interpolated data")
    curr_i = 0
    for k in tqdm(range(len(t_int))):
        if t_int[k] >= curr_i * TIMESTEP:
            curr_i += 1
            curr_row = np.hstack([
                t_int[k], position[k], acceleration[k], jerk[k]])
            output = np.vstack([output, curr_row])

    output = output[1:]
    output = np.round_(output, decimals=5)
    np.savetxt(args.out_path, output, fmt="%10.5f", delimiter='\t')


def plot_traj(position):
    position = position.T
    for p in range(len(position)):
        plt.plot(position[p], label=f"joint {p}")
    plt.title("Planned Trajectory")
    plt.xlabel("Time (steps)")
    plt.ylabel("Angle (rad)")
    ax = plt.legend()
    plt.show()


def do_fft():
    sh_filtered = fourier_filter(sh, 20)
    el_filtered = fourier_filter(el, 20)
    wr_filtered = fourier_filter(wr, 20)
    rt_filtered = fourier_filter(rt, 20)

    position = np.vstack([
        np.pi - np.sin(np.arange(len(t)) * 2 * 5 *
                       np.pi / len(t)) * 0.8,  # shoulder rotation
        sh_filtered * 1,  # shoulder abduction
        el_filtered * 1,  # elbow angle
        wr_filtered,  # wrist flexion
        # wrist yaw (currently unused due to self-collisions)
        np.ones_like(t) * np.pi / 2,
        rt_filtered * 1 - np.pi / 2,  # wrist roll
    ]).T

    plot_traj(position)
    save_traj(position)


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
    cc = np.sum(cc, axis=0)
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
                          ]).T

    plot_traj(position)
    position = linear_interp(position, int(1 / FRAMERATE / TIMESTEP))
    t_int = linear_interp(t[np.newaxis].T, int(1 / FRAMERATE / TIMESTEP))

    velocity = num_deriv(position, TIMESTEP)
    acceleration = num_deriv(velocity, TIMESTEP)
    jerk = num_deriv(acceleration, TIMESTEP)

    output = np.hstack([t_int, position, velocity, acceleration, jerk])
    output = np.round_(output, decimals=5)
    np.savetxt(args.out_path, output, fmt="%10.5f", delimiter='\t')


def top_n(arr, n=10):
    arr_d = np.fft.rfft(arr)
    arr_d[n:] = 0
    return np.fft.irfft(arr_d)


def remove_cluster(arr):
    out = []
    out.append(arr[0])

    for k in arr:
        if np.abs(k-out[-1]) > 30:
            out.append(k)

    return out


def do_decomp():
    global t

    sh_filtered = fourier_filter(sh, 20)
    el_filtered = fourier_filter(el, 20)
    wr_filtered = fourier_filter(wr, 20)
    rt_filtered = fourier_filter(rt, 20)

    sh_d = np.abs(np.gradient(sh_filtered, axis=0))
    el_d = np.abs(np.gradient(el_filtered, axis=0))
    wr_d = np.abs(np.gradient(wr_filtered, axis=0))

    sh_rest = top_n(sh_filtered, n=3)
    el_rest = top_n(el_filtered, n=5)
    wr_rest = top_n(wr_filtered, n=7)

    shr_d = np.abs(np.gradient(sh_rest, axis=0))
    elr_d = np.abs(np.gradient(el_rest, axis=0))
    wrr_d = np.abs(np.gradient(wr_rest, axis=0))

    zeros = np.where(sh_d < 0.005)[0]
    full_z = [0]
    for z in zeros:
        if el_d[z] + wr_d[z] < 0.01:
            full_z += [z]
    full_z = remove_cluster(full_z)

    zeros = np.where(shr_d < 0.001)[0]
    rest_z = [0]
    for z in zeros:
        if elr_d[z] + wrr_d[z] < 0.002:
            rest_z += [z]
    rest_z = remove_cluster(rest_z)

    print(full_z, "\n", rest_z)

    proto_pos = np.zeros((3, 1))
    for i in range(5, 6):
        low = np.random.choice(rest_z[:4])
        high = np.random.choice(rest_z[-4:])

        rest = sh_rest[low:high] - \
            sh_rest[low]+sh_filtered[full_z[i]]
        final_sh = np.hstack(
            (sh_filtered[:full_z[i]], rest, sh_filtered[full_z[i]:]-sh_filtered[full_z[i]]+rest[-1]))

        rest = el_rest[low:high] - \
            el_rest[low]+el_filtered[full_z[i]]
        final_el = np.hstack(
            (el_filtered[:full_z[i]], rest, el_filtered[full_z[i]:]-el_filtered[full_z[i]]+rest[-1]))

        rest = wr_rest[low:high] - \
            wr_rest[low]+wr_filtered[full_z[i]]
        final_wr = np.hstack(
            (wr_filtered[:full_z[i]], rest, wr_filtered[full_z[i]:]-wr_filtered[full_z[i]]+rest[-1]))

        curr_out = np.vstack([final_sh, final_el, final_wr])

        proto_pos = np.hstack([proto_pos, curr_out])

    t = np.arange(len(proto_pos[0])) / FRAMERATE

    position = np.vstack([
        np.pi - np.sin(np.arange(len(t)) * 2 * 5 *
                       np.pi / len(t)) * 0.8,  # shoulder rotation
        proto_pos,
        # wrist yaw (currently unused due to self-collisions)
        np.ones_like(t) * np.pi / 2,
        np.ones_like(t) * np.pi / 2,  # wrist roll
    ]).T

    plot_traj(position)
    save_traj(position)


# Rest of program
if (args.filter == "fft"):
    do_fft()
elif (args.filter == "cc"):
    do_cc()
elif (args.filter == "decomp"):
    do_decomp()
else:
    print("Option not implemented yet")
