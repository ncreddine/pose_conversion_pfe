import numpy as np
from tqdm import tqdm
import pathlib
import h5py
from matplotlib import pyplot as plt

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 6, 10, 11, 12, 13, 10, 15, 16, 17, 10, 19, 20, 21, 10, 23, 24, 25, 10, 27, 28, 29, 3, 31, 32, 33, 34, 31, 36, 37, 38, 31, 40, 41, 42, 31, 44, 45, 46, 31, 48, 49, 50] # colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
default_skeleton = list(zip(range(len(PARENTS)), PARENTS))[1:]

def draw_poses(ax, ar, data, default_skeleton, color = "blue"):
    print(data.shape)
    data = data.reshape(52, 2)
    artist = ax.scatter(*data.T, s = 10, color = color)
    ar.append(artist)
    for origin, end in default_skeleton :
        x, y   = data[end] 
        xx, yy = data[origin]
        x = x  ; y = y 
        xx = xx; yy = yy
        line, = ax.plot([x, xx],[y, yy], color = color,  linewidth = 2)
        ar.append(line)
    return ax, ar

PATH_TO_H5 = pathlib.Path("/media/holdee/MyPassport/dataset/dev/")

pose3d = []
pose2d = []

# for ind, file_ in tqdm(enumerate(PATH_TO_H5.glob('*/*')), ncols = 100) :
#     f = h5py.File(file_, "r")
#     missing_intervals = np.array(f['missing_intervals'])
#     p2 = np.array(f['pose2d'])
#     p3 = np.array(f['pose3d'])

#     p2 = np.array([ p2[52*i : 52*(i+1), :].flatten() for i  in range(len(p2) // 52) ])
#     p3 = np.array([ p3[52*i : 52*(i+1), :].flatten() for i  in range(len(p3) // 52) ])
#     f.close()

#     h5f = h5py.File(file_ , 'w')
#     h5f.create_dataset('missing_intervals', data = missing_intervals)
#     h5f.create_dataset('pose2d', data = p2)
#     h5f.create_dataset('pose3d', data = p3)
#     h5f.close()

for ind, file_ in tqdm(enumerate(PATH_TO_H5.glob('*/*')), ncols = 100) :
    f = h5py.File(file_, "r")
    pose2d.append(np.array(f['pose2d']))
    pose3d.append(np.array(f['pose3d']))

pose2d = np.vstack(pose2d)
pose3d = np.vstack(pose3d)

np.save("./pose2D_test.npy", pose2d)
np.save("./pose3D_test.npy", pose3d)

# i = 10000

ar = []
fig, ax = plt.subplots( figsize = (10,6))
ax, ar = draw_poses(ax, ar, pose2d[i], default_skeleton)
plt.show()

