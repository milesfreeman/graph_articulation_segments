import open3d as o3d
import scipy.io as matlab
import numpy as np 
import sys
import copy

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

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def loadit(fname):
    global colordicc
    contents = matlab.loadmat(fname)
    depthImg = contents['depthImg']
    labels = contents['labels']
    # print(np.unique(labels))
    
    theta_v = np.deg2rad(45.6)
    theta_h = np.deg2rad(58.5)
    pcd = []
    instances = []
    colors = []
    nearest = np.min(depthImg[np.nonzero(depthImg)])
    furthest = np.max(depthImg)
    depthImg /= furthest
    for i in range(480):
        for j in range(640):
            if depthImg[i][j] < (200 / furthest) or depthImg[i][j] > (500 / furthest): continue
            x = depthImg[i][j] / np.tan(0.5*(np.pi - theta_h) + (theta_h*(j+1)/640))
            y = depthImg[i][j] * np.tan((2*np.pi) - (0.5*theta_v) + ((i+1)*theta_v/480))
            z = depthImg[i][j] - (300/furthest)
            pcd.append([x,y,z])
            instances.append(labels[i][j])
            colors.append(colordicc[labels[i][j]])
    o_pcd = o3d.geometry.PointCloud()
    o_pcd.points = o3d.utility.Vector3dVector(pcd)
    o_pcd.colors = o3d.utility.Vector3dVector(colors)
    return o_pcd

def main():
    source = sys.argv[1]
    target = sys.argv[2]
    target = loadit(target)
    source = loadit(source)
    rf = -np.pi / 4
    current_transformation = [[np.cos(rf),  0, np.sin(rf), 0],
                   [0,        1, 0,       0],   
                   [-1*np.sin(rf), 0, np.cos(rf), 0],
                   [0,        0, 0,       1]]
    # current_transformation=np.identity(4)
    draw_registration_result_original_color(source, target,
                                            current_transformation)
    result_icp = o3d.registration.registration_icp(
        source, target, 0.02, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)

    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)
