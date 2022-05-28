import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# C0_05 IS IMG_4014
# C0_06 IS IMG_4014
# C0_07 is IMG_4014
# C0_09 IS IMG_4015
# C0_10 is IMG_4015 TWICE
# C0_11 is IMG_4015 SLOW
# C0_13 is IMG_4013 TWICE
# C0_14 is IMG_4010 TWICE
FRAMERATE = 30  # FPS in original video, should be 30

"""
Loading C005
"""
print("Processing C0_05")
with open("saves/C0_05", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4014.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)
combined_data = np.zeros(266+6)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

combined_data = combined_data[1:]


"""
Loading C006
"""
print("Processing C0_06")
with open("saves/C0_06", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4014.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C007
"""
print("Processing C0_07")

with open("saves/C0_07", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4014.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C009
"""
print("Processing C0_09")

with open("saves/C0_09", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4015.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C010
"""
print("Processing C0_10")
# First segment 0-38 seconds
# Second segment 1:57 to end
with open("saves/C0_10", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4015.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(38*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+57)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C013
"""
print("Processing C0_13")
# First segment 0-1:11
# Second segment 1:26 to end
with open("saves/C0_13", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4013.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range((60+11)*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+26)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C014
"""
print("Processing C0_14")
# First segment 0-1:11
# Second segment 1:26 to end
with open("saves/C0_14", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4010.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range((60+12)*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+14)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

print(f"Final shape is {combined_data.shape}")

with open(os.path.join("saves", "neural"), 'wb') as f:
    np.save(f, combined_data)
