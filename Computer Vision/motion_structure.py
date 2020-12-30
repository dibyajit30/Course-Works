# -*- coding: utf-8 -*-
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

matrix_file = 'sfm_points.mat'

def visualize(images):
    fig = plt.figure() 
    ax = plt.axes(projection ='3d') 
    if len(images.shape) == 2:
        ax.plot3D(images[0], images[1], 0, 'green')
    else:
        ax.plot3D(images[0], images[1], images[2], 'green')
    ax.set_title('Structure') 
    plt.show()

def values(matrix):
    w = []
    t = []
    for i in range(matrix.shape[1]):
        image = matrix[:, i, :]
        cx, cy = image[0].mean(), image[1].mean()
        t.append([cx, cy])
        centered_points = []
        for j in range(matrix.shape[2]):
            centered_x = image[0][j] - cx
            centered_y = image[1][j] - cy
            #centered_points.append([centered_x, centered_y])
            centered_points.append(centered_x)
            centered_points.append(centered_y)
        w.append(centered_points)
    
    w = np.array(w)
    w = w.reshape((w.shape[1], w.shape[0]))
    
    u, s, vh = np.linalg.svd(w, full_matrices=True)
    
    m = np.matmul(u[:, :3], s[:3])
    world_points = vh[:3]
    return world_points, m, t

def structure(matrix_file=matrix_file):
    matrix = sio.loadmat(matrix_file)['image_points']
    visualize(matrix[:, 1, :])
    world_points, m, t = values(matrix)
    visualize(world_points[:10])
    print('Camera matrix for first camera: ', m)
    print('Translation for first image: ', t[0])
    return world_points, m, t

if __name__ == '__main__':
    structure(matrix_file=matrix_file)