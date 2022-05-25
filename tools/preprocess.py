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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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

for n in range(len(raw_list)):
    frame_dict = []
    d = raw_list[n]["people"][0]
    body = d["pose_keypoints_2d"]
    face = d["face_keypoints_2d"]
    hand_l = d["hand_left_keypoints_2d"]
    hand_r = d["hand_right_keypoints_2d"]

    for i in range(24):
        x = body[i*3]
        y = body[i*3+1]
        frame_dict.append(x)
        frame_dict.append(y)

    for i in range(20):
        x = hand_r[i*3]
        y = hand_r[i*3+1]
        frame_dict.append(x)
        frame_dict.append(y)

        x = hand_l[i*3]
        y = hand_l[i*3+1]
        frame_dict.append(x)
        frame_dict.append(y)

    # for i in range(69):
    #     x = face[i*3]
    #     y = face[i*3+1]
    #     frame_dict.append(x)
    #     frame_dict.append(y)

    all_frames.append(np.array(frame_dict))

prev_vector = all_frames[0]

l = len(prev_vector)
print("Processing")
for n in range(1, len(raw_list)):
    curr_vector = all_frames[n]
    print(f"Processing {n}")
    for k in tqdm(range(l//2)):
        # If we have a point that is not here, then first find the last known location
        if (curr_vector[k*2], curr_vector[k*2+1]) == (0, 0):

            prev = (prev_vector[k*2], prev_vector[k*2+1])
            following = prev
            for tmp in range(n+1,  len(raw_list)):
                candidate_vector = all_frames[tmp]
                # We have found a suitable candidate
                if (candidate_vector[k*2], candidate_vector[k*2+1]) != (0, 0):
                    print(f"found hit for {n}, {k} at {tmp}")
                    following = (
                        candidate_vector[k*2], candidate_vector[k*2+1])

                    if prev == (0, 0):
                        prev = following

                    material = np.linspace(prev, following, tmp-n)
                    # If we are interpolating over mutliple steps, fill all the steps
                    for i in range(len(material)):
                        (all_frames[n+i][k*2], all_frames[n+i]
                            [k*2+1]) = material[i]

                    break

                # If we have not, keep looping

        # Otherwise, update last known location
        else:
            (prev_vector[k*2], prev_vector[k*2+1]
             ) = (curr_vector[k*2], curr_vector[k*2+1])


all_frames = np.array(all_frames)
with open(os.path.join("saves", args.file_path.split('/')[-1]), 'wb') as f:
    np.save(f, np.array(all_frames))
