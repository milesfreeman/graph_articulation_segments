import open3d as o3d
import scipy.io as matlab
from scipy.spatial import distance
import numpy as np 
import sys
import copy
from multiprocessing import Pool
import time
import pickle as pk

colordicc = {
    0: [0,0,0],
    1: [1,0,0],
    2: [1,0.5,0],
    3: [1,1,0],
    4: [0.5,1,0],
    5: [0.5,0.5,0],
    6: [0,1,0],
    7: [0,1,0.5],
    8: [0,0.5,1],
    9: [0,1,1],
    10: [0,0.5,0.5],
    11: [0,0,1],
    12: [1,0,1],
    13: [0.5,0,0.5],
    14: [0.5,0,1],
    15: [1,0,0.5],
    16: [0.5,0.5,0.5]
}

def loadit(fname, theta, phi, color):
    global colordicc
    depthImg = matlab.loadmat(fname)['DpthImg']
    theta_v = np.deg2rad(45.6)
    theta_h = np.deg2rad(58.5)
    pcd = []
    instances = []
    colors = []
    for i in range(480):
        for j in range(640):
            if depthImg[i][j] < 200 or depthImg[i][j] > 350: continue
            x = depthImg[i][j] / np.tan(0.5*(np.pi - theta_h) + (theta_h*(j+1)/640))
            y = depthImg[i][j] * np.tan((2*np.pi) - (0.5*theta_v) + ((i+1)*theta_v/480))
            z = depthImg[i][j] - 300
            pcd.append([x/100,y/100,z/100])
        
    o_pcd = o3d.geometry.PointCloud()
    o_pcd.points = o3d.utility.Vector3dVector(pcd)
    o_pcd.paint_uniform_color(color)

    R_x = np.array([[1, 0,      0,       0],
                    [0, np.cos(phi), -np.sin(phi), 0],
                    [0, np.sin(phi), np.cos(phi),  0],
                    [0, 0,      0,       1]])
    
    R_y = np.array([[np.cos(theta + np.pi),  0, np.sin(theta + np.pi), 0],
                    [0,      1, 0,     0],
                    [-np.sin(theta + np.pi), 0, np.cos(theta + np.pi), 0],
                    [0,      0, 0,     1]])
    R_z = np.array([[-np.cos(np.pi), -np.sin(np.pi), 0, 0],
                    [np.sin(np.pi),  np.cos(np.pi), 0, 0],
                    [0,  0, 1, 0],
                    [0,  0, 0, 1]])
    o_pcd.transform(R_x)
    o_pcd.transform(R_y)
    o_pcd.transform(R_z)
    return o_pcd

def point_target_match(targets, labels, threshold=0.09):

    def callable(xyz):
        for i in range(len(targets)):
            if distance.euclidean(xyz, targets[i]) < threshold:
                return xyz, labels[i]
    
    return callable
    


def main():
    
    L=loadit('45243_6L.mat', 7*np.pi/4, 0, [1,0,0])
    L = L.voxel_down_sample(0.05)

    R=loadit('45243_6R.mat', np.pi/4, 0, [0,1,0])
    R = R.voxel_down_sample(0.05)

    C=loadit('45243_6C.mat', 0, -np.pi/6, [0,0,1])
    C = C.voxel_down_sample(0.05)
    
    with open('45243_6PCD.txt', 'r') as f:
        lines = f.readlines()
    GT_PCD = []
    for line in lines:
        xyz = line.split()
        GT_PCD.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    with open('45243_6LABEL.txt', 'r') as f:
        lines = f.readlines()
    GT_COLORS = []
    GT_LABELS = []
    for line in lines:
        it = line.split()
        GT_COLORS.append(colordicc[int(it[0])])
        GT_LABELS.append(int(it[0]))

    GT = o3d.geometry.PointCloud()
    GT.points = o3d.utility.Vector3dVector(GT_PCD)
    GT.colors = o3d.utility.Vector3dVector(GT_COLORS)
    o3d.visualization.draw_geometries([GT])

    func = point_target_match(GT_PCD, GT_LABELS, threshold=0.025)
    # print('Est. time: 30 min')
    # # # p = Pool(4)
    # start = time.time()
    # centre = list(map(func, list(np.asarray(C.points))))
    # centre = list(filter(lambda x: x is not None, centre))
    # end = time.time()
    # print(f"Finished: {end-start} seconds")
    # # C_pts = list(list(zip(*centre))[0])
    # # C_labels = list(list(zip(*centre))[1])
    with open('Centre_fin.pkl', 'rb') as f:
       centre =  pk.load(f)
    C2 = o3d.geometry.PointCloud()
    C_pts = list(list(zip(*centre))[0])
    C_labels = list(list(zip(*centre))[1])
    C2.points = o3d.utility.Vector3dVector(C_pts)
    colors = list(map(lambda x: colordicc[x], C_labels))
    C2.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([C2])
    # print("Est. time: 21 min")
    # start = time.time()
    # left = list(map(func, list(np.asarray(L.points))))
    # left = list(filter(lambda x: x is not None, left))
    # end = time.time()
    # print(f"Finished Left: {(end-start) / 60} minutes")
    with open('Left_fin.pkl', 'rb') as f:
       left =  pk.load(f)
    L2 = o3d.geometry.PointCloud()
    L_pts = list(list(zip(*left))[0])
    L_labels = list(list(zip(*left))[1])
    L2.points = o3d.utility.Vector3dVector(L_pts)
    colors = list(map(lambda x: colordicc[x], L_labels))
    L2.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([L2])
    
    # start = time.time()
    # right = list(map(func, list(np.asarray(R.points))))
    # right = list(filter(lambda x: x is not None, right))
    # end = time.time()
    # print(f"Finished Right: {(end-start) / 60} minutes")
    with open('Right_fin.pkl', 'rb') as f:
       right =  pk.load(f)
    R2 = o3d.geometry.PointCloud()
    R_pts = list(list(zip(*right))[0])
    R_labels = list(list(zip(*right))[1])
    R2.points = o3d.utility.Vector3dVector(R_pts)
    colors = list(map(lambda x: colordicc[x], R_labels))
    R2.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([C2])
    o3d.visualization.draw_geometries([L2])
    o3d.visualization.draw_geometries([R2])

    o3d.visualization.draw_geometries([C2,L2,R2])
   



main()