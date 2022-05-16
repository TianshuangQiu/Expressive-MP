from cProfile import label
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
parser.add_argument("generate_vis", type=int,
                    help="Should I save a gif?")
args = parser.parse_args()


TIMESTEP = 0.001  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 30  # FPS in original video, should be 30
FRAMES = args.n
prev_hand_kp = [0, 0]
prev_thumb_kp = [0, 0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


def extract_angles(chest, shoulder, elbow, wrist, fingertip, thumbtip):
    """
    Given the positions of the shoulder, elbow, wrist,
    and fingertip, extract the three desired angles.
    """
    xdiff0 = shoulder[0] - chest[0]
    ydiff0 = shoulder[1] - chest[1]
    theta0 = np.arctan2(ydiff0, xdiff0)

    xdiff1 = elbow[0] - shoulder[0]
    ydiff1 = elbow[1] - shoulder[1]
    theta1 = np.arctan2(ydiff1, xdiff1)

    xdiff2 = wrist[0] - elbow[0]
    ydiff2 = wrist[1] - elbow[1]
    theta2 = np.arctan2(ydiff2, xdiff2)

    xdiff3 = fingertip[0] - wrist[0]
    ydiff3 = fingertip[1] - wrist[1]
    theta3 = np.arctan2(ydiff3, xdiff3)

    xdiff4 = thumbtip[0] - wrist[0]
    ydiff4 = thumbtip[1] - wrist[1]
    theta4 = np.arctan2(ydiff4, xdiff4)

    return [theta0, theta1, theta2, theta3, theta4]


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


def get_thumb_kpt(d):
    """
    Get the thumb detection in order to find the rotation angle of the wrist.
    """
    global prev_thumb_kp
    thumb = None

    keypoints = d['hand_right_keypoints_2d']
    for i in [4, 3, 2, 1]:
        p = keypoints[3 * i: 3 * i + 2]
        if p[0] != 0 and p[1] != 0:
            thumb = p
            break
    if thumb == None:
        return prev_thumb_kp
    else:
        return thumb


def linear_interp(array, num):
    interpolated = array[0]
    for i in range(len(array) - 1):
        interp = np.linspace(array[i], array[i + 1], num)
        interpolated = np.vstack([interpolated, interp])

    return interpolated[1:]


theta_list = []
all_frames = {}

for n in range(FRAMES):
    d = readfile(n)["people"][0]
    series = d['pose_keypoints_2d']
    body_dict = {}

    hand_pt = get_hand_kpt(d)
    thumb_pt = get_thumb_kpt(d)

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
        thumb_pt,
        hand_pt)
    theta_list.append(thetas)

    all_frames[n] = body_dict


hand_length = -1
head_size = -1


def animate(i):
    global hand_length
    global head_size
    frame_dict = all_frames[i]

    if not(frame_dict[1] == (0, 0) or frame_dict[2] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[2][0])
        y = np.linspace(frame_dict[1][1], frame_dict[2][1])
        shoulder1.set_data(x, y)

    if not(frame_dict[1] == (0, 0) or frame_dict[5] == (0, 0)):
        x = np.linspace(frame_dict[1][0], frame_dict[5][0])
        y = np.linspace(frame_dict[1][1], frame_dict[5][1])
        shoulder2.set_data(x, y)

    # if not(frame_dict[0] == (0, 0) or frame_dict[14] == (0, 0)):
    #     x = np.linspace(frame_dict[0][0], frame_dict[15][0])
    #     y = np.linspace(frame_dict[0][1], frame_dict[15][1])
    #     head1.set_data(x, y)

    # if not(frame_dict[0] == (0, 0) or frame_dict[15] == (0, 0)):
    #     x = np.linspace(frame_dict[0][0], frame_dict[16][0])
    #     y = np.linspace(frame_dict[0][1], frame_dict[16][1])
    #     head2.set_data(x, y)

    if head_size < 70:
        head_size = np.max([np.linalg.norm(np.array(frame_dict[0])-np.array(frame_dict[15])),
                            np.linalg.norm(np.array(frame_dict[0])-np.array(frame_dict[16]))])

    bozi = np.array(frame_dict[0])-np.array(frame_dict[1])
    center = np.array(frame_dict[0])

    circ.set_data(1.5*head_size*np.cos(np.linspace(0, 2*np.pi))+center[0],
                  1.5*head_size*np.sin(np.linspace(0, 2*np.pi))+center[1])

    if not(frame_dict[0] == (0, 0) or frame_dict[1] == (0, 0)):
        bozi = bozi/np.linalg.norm(bozi)*(np.linalg.norm(bozi)-1.5*head_size)
        neck.set_data(np.linspace(frame_dict[1][0], frame_dict[1][0]+bozi[0]),
                      np.linspace(frame_dict[1][1], frame_dict[1][1]+bozi[1]))

    # if not(frame_dict[14] == (0, 0) or frame_dict[16] == (0, 0)):
    #     x = np.linspace(frame_dict[16][0], frame_dict[18][0])
    #     y = np.linspace(frame_dict[16][1], frame_dict[18][1])
    #     head3.set_data(x, y)

    # if not(frame_dict[15] == (0, 0) or frame_dict[17] == (0, 0)):
    #     x = np.linspace(frame_dict[15][0], frame_dict[17][0])
    #     y = np.linspace(frame_dict[15][1], frame_dict[17][1])
    #     head4.set_data(x, y)

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

    if hand_length < 70:
        hand_length = np.linalg.norm(np.array(frame_dict[25][0]) -
                                     np.array(frame_dict[4][0]))

    if not(frame_dict[4] == (0, 0) or frame_dict[25] == (0, 0)):
        hand_vec = np.array(frame_dict[25]) - np.array(frame_dict[4])
        hand_vec = hand_vec/np.linalg.norm(hand_vec)*hand_length
        hand.set_data(np.linspace(frame_dict[4][0], frame_dict[4][0]+hand_vec[0]),
                      np.linspace(frame_dict[4][1], frame_dict[4][1]+hand_vec[1]))

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

    base = np.array([0, 0])
    tip = np.array([np.cos(sh[i] + np.pi), np.sin(sh[i] + np.pi)])
    dat = np.linspace(base, tip)
    segment0.set_data(dat.T[0], dat.T[1])

    base = tip
    tip = np.array([np.cos(el[i]+sh[i] + np.pi),
                   np.sin(el[i]+sh[i] + np.pi)]) * 2 + tip
    dat = np.linspace(base, tip)
    segment1.set_data(dat.T[0], dat.T[1])

    base = tip
    tip = np.array([np.cos(wr[i]+el[i]+sh[i] + np.pi),
                   np.sin(wr[i]+el[i]+sh[i] + np.pi)]) + tip
    dat = np.linspace(base, tip)
    segment2.set_data(dat.T[0], dat.T[1])

    return shoulder1, shoulder2, neck, head1, head2, head3,\
        head4, elbow1, elbow2, chest, wrist1, wrist2, hand,\
        hip1, hip2, knee1, knee2, ankle1, ankle2, segment0,\
        segment1, segment2, circ


tl = np.array(theta_list)
tl[tl < 0] = tl[tl < 0] + 2 * np.pi

for i in range(len(tl) - 1):
    for j in range(len(tl[i])):
        if np.abs(tl[i, j] - tl[i + 1, j]) > np.abs(tl[i, j] - tl[i + 1, j] - 2 * np.pi):
            tl[i + 1, j] += 2 * np.pi
        if np.abs(tl[i, j] - tl[i + 1, j]) > np.abs(tl[i, j] - tl[i + 1, j] + 2 * np.pi):
            tl[i + 1, j] -= 2 * np.pi


kernel = np.array([1, 2, 4, 6, 10, 14, 17, 19, 17, 14, 10, 6, 4, 2, 1])
# kernel = np.ones(9)
kernel = kernel / np.sum(kernel)

smoothed_thetas = np.vstack([np.convolve(tl[:, 0], kernel, mode='same'),
                             np.convolve(tl[:, 1], kernel, mode='same'),
                             np.convolve(tl[:, 2], kernel, mode='same'),
                             np.convolve(tl[:, 3], kernel, mode='same'),
                             np.convolve(tl[:, 4], kernel, mode='same')])

data = smoothed_thetas[:]
t = np.arange(len(data[0])) / FRAMERATE

sh = data[0] - data[1] - np.pi / 2
el = data[1] - data[2]
wr = data[2] - data[3]
rt = data[3] - data[4]


rt = sigmoid(rt / np.max(np.abs(rt)) * 3) * np.pi / 2


fig, ax = plt.subplots()
ax.set_ylim([-5, 6])
ax.plot(sh, label="sh")
ax.plot(el, label="el")
ax.plot(wr, label="wr")
ax.plot(rt, label="rt")
ax.legend()
ax.set_title("Filtered Signal")
ax.set_xlabel("Frames")
ax.set_ylabel("Radians")
plt.show()


if (args.generate_vis):
    fig, (ax, fax) = plt.subplots(1, 2, figsize=(
        20, 10), sharex=False, sharey=False)
    ax.set_ylim([0, 2000])
    ax.set_xlim([1000, 3000])

    elbow1, = ax.plot([], [])
    wrist1, = ax.plot([], [])
    hand, = ax.plot([], [])

    shoulder1, = ax.plot([], [])
    shoulder2, = ax.plot([], [])
    neck, = ax.plot([], [])
    head1, = ax.plot([], [])
    head2, = ax.plot([], [])
    head3, = ax.plot([], [])
    head4, = ax.plot([], [])
    elbow2, = ax.plot([], [])
    wrist2, = ax.plot([], [])
    circ, = ax.plot([], [])
    chest, = ax.plot([], [])
    hip1, = ax.plot([], [])
    hip2, = ax.plot([], [])
    knee1, = ax.plot([], [])
    knee2, = ax.plot([], [])
    ankle1, = ax.plot([], [])
    ankle2, = ax.plot([], [])

    fax.set_ylim([-5, 5])
    fax.set_xlim([-5, 5])
    segment0, = fax.plot([], [])
    segment1, = fax.plot([], [])
    segment2, = fax.plot([], [])
    fax.axes.xaxis.set_visible(False)
    fax.axes.yaxis.set_visible(False)
    fax.set_title("Joint Angle Reconstruction")

    ax.invert_yaxis()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title("OpenPose Waypoints")
    anim = FuncAnimation(
        fig,
        animate,
        frames=FRAMES,
        interval=20,
        blit=True,
    )
    anim.save(args.file_path.split("/")[-1] + ".gif")

data_stack = np.vstack([sh, el, wr, rt])
with open("saves/"+args.file_path.split("/")[-1], "wb") as f:
    np.save(f, data_stack)
