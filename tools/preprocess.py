from cProfile import label
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description='Parse data files')
parser.add_argument('file_path', type=str,
                    help='Where is the json folder stored at?')
args = parser.parse_args()

TIMESTEP = 0.001
FRAMERATE = 30


def readfile():
    """
    Reads all files that end with .json and sorts it
    """

    # Processing inputs
    files = []
    for _, _, fnames in os.walk(args.file_path):
        for f in fnames:
            if "json" in f:
                files.append(os.path.join(args.file_path, f))

    files.sort()
    lst = []
    for f in tqdm(files):
        item = pd.read_json(f)
        lst.append(item)
    return lst


all_frames = []
print("Reading files")

raw_list = readfile()
confidence_arr = []

for n in range(len(raw_list)):
    frame_dict = []
    frame_confidence = []
    d = raw_list[n]["people"][0]
    body = d["pose_keypoints_2d"]
    face = d["face_keypoints_2d"]
    hand_l = d["hand_left_keypoints_2d"]
    hand_r = d["hand_right_keypoints_2d"]

    for i in range(24):
        x = body[i*3]
        y = body[i*3+1]
        confidence = body[i*3+2]
        frame_dict.append(x)
        frame_dict.append(y)
        frame_confidence.append(confidence)

    for i in range(20):
        x = hand_r[i*3]
        y = hand_r[i*3+1]
        confidence = hand_r[i*3+2]
        frame_dict.append(x)
        frame_dict.append(y)
        frame_confidence.append(confidence)

        x = hand_l[i*3]
        y = hand_l[i*3+1]
        confidence = hand_l[i*3+2]
        frame_dict.append(x)
        frame_dict.append(y)
        frame_confidence.append(confidence)

    for i in range(69):
        x = face[i*3]
        y = face[i*3+1]
        confidence = face[i*3+2]
        frame_dict.append(x)
        frame_dict.append(y)
        frame_confidence.append(confidence)

    confidence_arr.append(np.array(frame_confidence))
    all_frames.append(frame_dict)

confidence_arr = np.array(confidence_arr).T
all_frames = np.array(all_frames).T


print("Processing")

for n in tqdm(range(1, len(all_frames))):
    curr_vector = all_frames[n]
    confidence_vector = confidence_arr[n//2]
    prev_val = curr_vector[0]
    # plt.plot(curr_vector, label="unfiltered")
    # plt.plot(confidence_vector*1000, label="conf")

    k = 1
    while k < len(curr_vector):
        # If we have a point that is not here, then first find the last known location
        if curr_vector[k] == 0 or confidence_vector[k] < 0.6:
            for tmp in range(k,  len(curr_vector)):

                candidate = curr_vector[tmp]
                candidate_confidence = confidence_vector[tmp]

                if candidate_confidence > 0.6 and candidate != 0:

                    # We have found a suitable candidate
                    filler = np.linspace(prev_val, candidate, tmp-k)
                    # If we are interpolating over mutliple steps, fill all the steps

                    for i in range(len(filler)):
                        all_frames[n][k+i] = filler[i]
                        curr_vector[k+i] = filler[i]
                        confidence_arr[n//2][k+i] = 0.6
                        confidence_vector[k+i] = 0.6

                    prev_val = candidate
                    k = tmp+1
                    break
                # If we have not, keep looping

                # Stops perma loop near the end
                loop_breaker = confidence_vector[tmp:]
                if len(loop_breaker[loop_breaker > 0.6]) == 0:
                    all_frames[n][k:] = prev_val
                    confidence_arr[n//2][tmp:] = 0.6
                    k = len(curr_vector)
                    break

                # Final check, just in case
                if len(loop_breaker) < 5:
                    all_frames[n][k:] = prev_val
                    confidence_arr[n//2][tmp:] = 0.6
                    k = len(curr_vector)
                    break

        # Otherwise, update last known location
        else:
            prev_val = curr_vector[k]
            k += 1

    # plt.plot(all_frames[n], label="filtered")
    # plt.legend()
    # plt.show()


# all_frames = np.vstack([all_frames., confidence_arr.])

# def fourier_filter(array, thresh):
#     fourier = np.fft.rfft(array)
#     fourier[np.abs(fourier) < thresh] = 0
#     filtered = np.fft.irfft(fourier)
#     return filtered


# print("Final smoothing")
# for i in tqdm(range(len(all_frames.T))):
#     pixl_val = all_frames.T[i]

#     ftr_val = fourier_filter(pixl_val, 5000)

#     if i in [4, 5, 6, 7, 8, 9, 10, 11]:
#         plt.plot(pixl_val)
#         plt.plot(ftr_val)
#         plt.show()
#     for j in range(len(ftr_val)):
#         all_frames[j][i] = ftr_val[j]


with open(os.path.join("saves", args.file_path.split('/')[-1]), 'wb') as f:
    np.save(f, all_frames.T)
