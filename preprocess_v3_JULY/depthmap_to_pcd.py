import open3d as o3d
import scipy.io as matlab
from scipy.spatial import distance
import numpy as np 
import sys
import copy
import time
import os
import pickle as pk

colordicc = {
    -1: [0.7,0.7,0.7],
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
    16: [0.5,0.5,0.5],
    17: [0.25,0.75,0.75],
    18: [0.75,0.25,0.25],
    19: [0.5,0.25,0.25],
    20: [0.25,0.5,0.75]

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
            if depthImg[i][j] < 200 or depthImg[i][j] > 400: continue
            x = depthImg[i][j] / np.tan(0.5*(np.pi - theta_h) + (theta_h*(j+1)/640))
            y = depthImg[i][j] * np.tan((2*np.pi) - (0.5*theta_v) + ((i+1)*theta_v/480))
            z = depthImg[i][j] - 300
            pcd.append([x/100,y/100,z/100])
        
    o_pcd = o3d.geometry.PointCloud()
    o_pcd.points = o3d.utility.Vector3dVector(pcd)
    # o3d.visualization.draw_geometries([o_pcd])
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

def point_target_match(targets, labels, threshold=0.05):

    def callable(xyz):
        for i in range(len(targets)):
            if distance.euclidean(xyz, targets[i]) < threshold: return xyz, labels[i]
        return xyz, -1
    return callable
    
def drop_points(pcd, n):
    pts = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)[0]
    k = np.random.randint(len(pts), size=n)
    pcd.points = o3d.utility.Vector3dVector(np.array(list(map(lambda x: pts[x], k))))
    pcd.paint_uniform_color(color)
    return pcd

def main():
    i = sys.argv[1]
    
    try:
        Rf=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Rf.F.mat', -7*np.pi/4, 0, [1,0,0])
    except FileNotFoundError:
        print(f"Missing: {i} left")
        exit(0)
    # L = L.voxel_down_sample(0.05)
    Rf = drop_points(Rf,2500)
   # o3d.visualization.draw_geometries([Rf])
    
    try:
        Lf=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Lf.F.mat', -np.pi/4, 0, [0,1,0])
    except FileNotFoundError:
        print(f"Missing: {i} right")
        exit(0)
    # R = R.voxel_down_sample(0.05)
    Lf = drop_points(Lf, 2500)
   # o3d.visualization.draw_geometries([Lf])
    
    try:
        Cu=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Cu.F.mat', 0, -np.pi/6, [0,0,1])
    except FileNotFoundError:
        print(f"Missing: {i} centre")
        exit(0)
    # C = C.voxel_down_sample(0.05)
    Cu = drop_points(Cu, 2500)
   # o3d.visualization.draw_geometries([Cu])

    try:
        Cd=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Cd.F.mat', 0, np.pi/6, [1,0,1])
    except FileNotFoundError:
        print(f"Missing: {i} centre under")
        exit(0)
    # C = C.voxel_down_sample(0.05)
    Cd = drop_points(Cd, 2500)
   # o3d.visualization.draw_geometries([Cd])

    try:
        Rb=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Rb.F.mat', -5*np.pi/4, 0, [1,1,0])
    except FileNotFoundError:
        print(f"Missing: {i} right back")
        exit(0)
    # L = L.voxel_down_sample(0.05)
    Rb = drop_points(Rb,2500)
   # o3d.visualization.draw_geometries([Rb])
    # o3d.visualization.draw_geometries([Rf, Rb, Cd, Cu, Lf])

    try:
        Lb=loadit(f'DATA/KINECT_OUT/chair/chair_{i}_Lb.F.mat', -3*np.pi/4, 0, [0,1,0])
    except FileNotFoundError:
        print(f"Missing: {i} left back")
        exit(0)
    # R = R.voxel_down_sample(0.05)
    Lf = drop_points(Lf, 2500)
   # o3d.visualization.draw_geometries([Lf])

    with open(f'DATA/PCD_XYZ/chair/chair_{i}.txt', 'r') as f:
        lines = f.readlines()
    GT_PCD = []
    for line in lines:
        xyz = line.split()
        GT_PCD.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    with open(f'DATA/PCD_LABEL/chair/chair_{i}.txt', 'r') as f:
        lines = f.readlines()
    GT_COLORS = []
    GT_LABELS = []
    for line in lines:
        it = line.split()
        if it[0] == '#' : continue
        GT_COLORS.append(colordicc[int(it[0])])
        GT_LABELS.append(int(it[0]))

    GT = o3d.geometry.PointCloud()
    GT.points = o3d.utility.Vector3dVector(GT_PCD)
    GT.colors = o3d.utility.Vector3dVector(GT_COLORS)
   # o3d.visualization.draw_geometries([GT])

    func = point_target_match(GT_PCD, GT_LABELS)
    
    print('Progress: [')
    centre_u = list(map(func, list(np.asarray(Cu.points))))
    print('-------1')
    centre_d = list(map(func, list(np.asarray(Cd.points))))
    print('-------2')
    left_f = list(map(func, list(np.asarray(Lf.points))))
    print('-------3')
    right_f = list(map(func, list(np.asarray(Rf.points))))
    print('-------4')
    right_b = list(map(func, list(np.asarray(Rb.points))))
    print('-------5')
    left_b = list(map(func, list(np.asarray(Lb.points))))
    print('-------6------] \n Complete!')

    Lf2 = o3d.geometry.PointCloud()
    Lf_pts = list(list(zip(*left_f))[0])
    Lf_labels = list(list(zip(*left_f))[1])
    Lf2.points = o3d.utility.Vector3dVector(Lf_pts)
    colors = list(map(lambda x: colordicc[x], Lf_labels))
    Lf2.colors = o3d.utility.Vector3dVector(colors)
   # o3d.visualization.draw_geometries([Lf2])

    Lb2 = o3d.geometry.PointCloud()
    Lb_pts = list(list(zip(*left_b))[0])
    Lb_labels = list(list(zip(*left_b))[1])
    Lb2.points = o3d.utility.Vector3dVector(Lb_pts)
    colors = list(map(lambda x: colordicc[x], Lb_labels))
    Lb2.colors = o3d.utility.Vector3dVector(colors)
   # o3d.visualization.draw_geometries([Lb2])

    Rf2 = o3d.geometry.PointCloud()
    Rf_pts = list(list(zip(*right_f))[0])
    Rf_labels = list(list(zip(*right_f))[1])
    Rf2.points = o3d.utility.Vector3dVector(Rf_pts)
    colors = list(map(lambda x: colordicc[x], Rf_labels))
    Rf2.colors = o3d.utility.Vector3dVector(colors)

    Rb2 = o3d.geometry.PointCloud()
    Rb_pts = list(list(zip(*right_b))[0])
    Rb_labels = list(list(zip(*right_b))[1])
    Rb2.points = o3d.utility.Vector3dVector(Rb_pts)
    colors = list(map(lambda x: colordicc[x], Rb_labels))
    Rb2.colors = o3d.utility.Vector3dVector(colors)

    Cu2 = o3d.geometry.PointCloud()
    Cu_pts = list(list(zip(*centre_u))[0])
    Cu_labels = list(list(zip(*centre_u))[1])
    Cu2.points = o3d.utility.Vector3dVector(Cu_pts)
    colors = list(map(lambda x: colordicc[x], Cu_labels))
    Cu2.colors = o3d.utility.Vector3dVector(colors)

    Cd2 = o3d.geometry.PointCloud()
    Cd_pts = list(list(zip(*centre_d))[0])
    Cd_labels = list(list(zip(*centre_d))[1])
    Cd2.points = o3d.utility.Vector3dVector(Cd_pts)
    colors = list(map(lambda x: colordicc[x], Cd_labels))
    Cd2.colors = o3d.utility.Vector3dVector(colors)

   # o3d.visualization.draw_geometries([Cu2,Cd2,Lb2,Rb2], window_name='View1')
   # o3d.visualization.draw_geometries([Lf2,Rf2,Lb2,Rb2], window_name='View2')
   # o3d.visualization.draw_geometries([Cu2,Cd2,Lf2,Rf2], window_name='View3')
   # o3d.visualization.draw_geometries([Cd2,Rf2,Lb2,Rb2], window_name='View4')
   # o3d.visualization.draw_geometries([Cu2,Rb2,Lf2,Rf2], window_name='View5')

    os.chdir('DATA/FINAL/chair')
    with open(f"chair_{i}_v1.pkl", 'wb') as f:
        pk.dump((Cu_pts + Cd_pts + Lb_pts + Rb_pts, Cu_labels + Cd_labels + Lb_labels + Rb_labels), f)
    with open(f"chair_{i}_v2.pkl", 'wb') as f:
        pk.dump((Lf_pts + Rf_pts + Lb_pts + Rb_pts, Lf_labels + Rf_labels + Lb_labels + Rb_labels), f)
    with open(f"chair_{i}_v3.pkl", 'wb') as f:
        pk.dump((Cu_pts + Cd_pts + Lf_pts + Rf_pts, Cu_labels + Cd_labels + Lf_labels + Rf_labels), f)
    with open(f"chair_{i}_v4.pkl", 'wb') as f:
        pk.dump((Cd_pts + Rf_pts + Lb_pts + Rb_pts, Cd_labels + Rf_labels + Lb_labels + Rb_labels), f)
    with open(f"chair_{i}_v5.pkl", 'wb') as f:
        pk.dump((Cu_pts + Rb_pts + Lf_pts + Rf_pts, Cu_labels + Rb_labels + Lf_labels + Rf_labels), f)

main()
