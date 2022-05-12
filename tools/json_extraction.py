import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser(
    description='Parse data files')
parser.add_argument('file_path', type=str,
                    help='Where is the json folder stored at?')
parser.add_argument('n', type=int,
                    help='How many frames are there?')
parser.add_argument("generate_vis", type=bool,
                    help="Should I save a gif?")
args = parser.parse_args()


TIMESTEP = 0.001  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 25  # FPS in original video, should be 25
FRAMES = args.n
prev_hand_kp = [0, 0]


def readfile(n):
    if n < 10:
        numstring = "000" + str(n)
    elif n < 100:
        numstring = "00" + str(n)
    elif n < 1000:
        numstring = "0" + str(n)
    else:
        numstring = str(n)

    filename = "/home/akita/autolab/Expressive-MP/waypoints/IMG_4015/IMG_4015_00000000" + \
        numstring + "_keypoints.json"
    # filename = args.file_path + numstring + "_keypoints.json"

    item = pd.read_json(filename)
    return item


def extract_angles(chest, shoulder, elbow, wrist, fingertip):
    """
    Given the positions of the shoulder, elbow, wrist, 
    and fingertip, extract the three desired angles.
    """
    v0 = shoulder - chest
    v1 = elbow - shoulder
    v2 = wrist - elbow
    v3 = fingertip - wrist

    theta1 = np.arccos(np.dot(v0, v1) / np.linalg.norm(v0)/np.linalg.norm(v1))
    theta2 = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1)/np.linalg.norm(v2))
    theta3 = np.arccos(np.dot(v2, v3) / np.linalg.norm(v2)/np.linalg.norm(v3))

    return [theta1, theta2, theta3]


def get_hand_kpt(d):
    """
    Hand keypoint detection is extremely noisy, so we do 
    the best we can and allow for lots of smoothing later.
    """
    global prev_hand_kp

    keypoints = d['hand_right_keypoints_2d']
    for i in [12, 16, 8, 20]:
        p = keypoints[3 * i: 3 * i + 2]
        if p[0] != 0 and p[1] != 0:
            prev_hand_kp = p
            return p
    return prev_hand_kp


def linear_interp(array, num):
    interpolated = array[0]
    for i in range(len(array) - 1):
        interp = np.linspace(array[i], array[i + 1], num)
        interpolated = np.vstack([interpolated, interp])

    return interpolated[1:]


if (args.generate_vis):
    theta_list = []
    all_frames = {}
    for n in range(FRAMES):
        d = readfile(n)["people"][0]
        series = d['pose_keypoints_2d']
        body_dict = {}

        hand_pt = get_hand_kpt(d)

        for i in range(24):
            x = series[i*3]
            y = series[i*3+1]
            body_dict[i] = (x, y)
        body_dict[25] = hand_pt

        thetas = extract_angles(
            np.array(body_dict[1]),
            np.array(body_dict[2]),
            np.array(body_dict[3]),
            np.array(body_dict[4]),
            hand_pt)
        theta_list.append(thetas)

        all_frames[n] = body_dict

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_ylim([0, 2000])
    ax.set_xlim([1000, 3000])

    shoulder1, = ax.plot([], [])
    shoulder2, = ax.plot([], [])
    neck, = ax.plot([], [])
    head1, = ax.plot([], [])
    head2, = ax.plot([], [])
    head3, = ax.plot([], [])
    head4, = ax.plot([], [])
    elbow1, = ax.plot([], [])
    elbow2, = ax.plot([], [])
    wrist1, = ax.plot([], [])
    wrist2, = ax.plot([], [])
    hand, = ax.plot([], [])
    chest, = ax.plot([], [])
    hip1, = ax.plot([], [])
    hip2, = ax.plot([], [])
    knee1, = ax.plot([], [])
    knee2, = ax.plot([], [])
    ankle1, = ax.plot([], [])
    ankle2, = ax.plot([], [])


def animate(i):
    frame_dict = all_frames[i]
    if not(frame_dict[1] == (0, 0) or frame_dict[2] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[2][0])
        y = np.linspace(frame_dict[1][1], frame_dict[2][1])
        shoulder1.set_data(x, y)

    if not(frame_dict[1] == (0, 0) or frame_dict[5] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[5][0])
        y = np.linspace(frame_dict[1][1], frame_dict[5][1])
        shoulder2.set_data(x, y)

    if not(frame_dict[0] == (0, 0) or frame_dict[1] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[0][0])
        y = np.linspace(frame_dict[1][1], frame_dict[0][1])
        neck.set_data(x, y)

    if not(frame_dict[0] == (0, 0) or frame_dict[14] == (0, 0)):
        x = np.linspace(frame_dict[0][0], frame_dict[15][0])
        y = np.linspace(frame_dict[0][1], frame_dict[15][1])
        head1.set_data(x, y)

    if not(frame_dict[14] == (0, 0) or frame_dict[16] == (0, 0)):
        x = np.linspace(frame_dict[16][0], frame_dict[18][0])
        y = np.linspace(frame_dict[16][1], frame_dict[18][1])
        head2.set_data(x, y)

    if not(frame_dict[0] == (0, 0) or frame_dict[15] == (0, 0)):
        x = np.linspace(frame_dict[0][0], frame_dict[16][0])
        y = np.linspace(frame_dict[0][1], frame_dict[16][1])
        head3.set_data(x, y)

    if not(frame_dict[15] == (0, 0) or frame_dict[17] == (0, 0)):
        x = np.linspace(frame_dict[15][0], frame_dict[17][0])
        y = np.linspace(frame_dict[15][1], frame_dict[17][1])
        head4.set_data(x, y)

    if not(frame_dict[2] == (0, 0) or frame_dict[3] == (0, 0)):
        x = np.linspace(frame_dict[2][0], frame_dict[3][0])
        y = np.linspace(frame_dict[2][1], frame_dict[3][1])
        elbow1.set_data(x, y)

    if not(frame_dict[5] == (0, 0) or frame_dict[6] == (0, 0)):
        x = np.linspace(frame_dict[5][0], frame_dict[6][0])
        y = np.linspace(frame_dict[5][1], frame_dict[6][1])
        elbow2.set_data(x, y)

    if not(frame_dict[3] == (0, 0) or frame_dict[4] == (0, 0)):
        x = np.linspace(frame_dict[3][0], frame_dict[4][0])
        y = np.linspace(frame_dict[3][1], frame_dict[4][1])
        wrist1.set_data(x, y)

    if not(frame_dict[6] == (0, 0) or frame_dict[7] == (0, 0)):
        x = np.linspace(frame_dict[6][0], frame_dict[7][0])
        y = np.linspace(frame_dict[6][1], frame_dict[7][1])
        wrist2.set_data(x, y)

    if not(frame_dict[4] == (0, 0) or frame_dict[25] == (0, 0)):
        x = np.linspace(frame_dict[4][0], frame_dict[25][0])
        y = np.linspace(frame_dict[4][1], frame_dict[25][1])
        hand.set_data(x, y)

    if not(frame_dict[1] == (0, 0) or frame_dict[8] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[8][0])
        y = np.linspace(frame_dict[1][1], frame_dict[8][1])
        chest.set_data(x, y)

    if not(frame_dict[8] == (0, 0) or frame_dict[9] == (0, 0)):
        x = np.linspace(frame_dict[8][0], frame_dict[9][0])
        y = np.linspace(frame_dict[8][1], frame_dict[9][1])
        hip1.set_data(x, y)

    if not(frame_dict[8] == (0, 0) or frame_dict[12] == (0, 0)):
        x = np.linspace(frame_dict[8][0], frame_dict[12][0])
        y = np.linspace(frame_dict[8][1], frame_dict[12][1])
        hip2.set_data(x, y)

    if not(frame_dict[9] == (0, 0) or frame_dict[10] == (0, 0)):
        x = np.linspace(frame_dict[9][0], frame_dict[10][0])
        y = np.linspace(frame_dict[9][1], frame_dict[10][1])
        knee1.set_data(x, y)

    if not(frame_dict[12] == (0, 0) or frame_dict[13] == (0, 0)):
        x = np.linspace(frame_dict[12][0], frame_dict[13][0])
        y = np.linspace(frame_dict[12][1], frame_dict[13][1])
        knee2.set_data(x, y)

    if not(frame_dict[10] == (0, 0) or frame_dict[11] == (0, 0)):
        x = np.linspace(frame_dict[10][0], frame_dict[11][0])
        y = np.linspace(frame_dict[10][1], frame_dict[11][1])
        ankle1.set_data(x, y)

    if not(frame_dict[13] == (0, 0) or frame_dict[14] == (0, 0)):
        x = np.linspace(frame_dict[13][0], frame_dict[14][0])
        y = np.linspace(frame_dict[13][1], frame_dict[14][1])
        ankle2.set_data(x, y)

    return shoulder1, shoulder2, neck, head1, head2, head3, head4, elbow1, elbow2, chest, wrist1, wrist2, hand, hip1, hip2, knee1, knee2, ankle1, ankle2


plt.gca().invert_yaxis()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
anim = FuncAnimation(
    fig,
    animate,
    frames=FRAMES,
    interval=20,
    blit=True,
)
anim.save(args.file_path.split("/")[-1] + ".gif")


tl = np.array(theta_list)
tl[tl < 0] = tl[tl < 0] + 2 * np.pi

kernel = np.array([1, 2, 4, 6, 10, 14, 17, 19, 17, 14, 10, 6, 4, 2, 1])
#kernel = np.ones(9)
kernel = kernel / np.sum(kernel)

smoothed_thetas = np.vstack([np.convolve(tl[:, 0], kernel, mode='same'),
                             np.convolve(tl[:, 1], kernel, mode='same'),
                             np.convolve(tl[:, 2], kernel, mode='same')])

data = smoothed_thetas[:]
t = np.arange(len(data[0])) / FRAMERATE

sh = 0 - data[0]
el = data[0] - data[1]
wr = data[1] - data[2]

data_stack = np.vstack([sh, el, wr])

with open("saves/IMG_4015", "wb") as f:
    np.save(f, data_stack)
