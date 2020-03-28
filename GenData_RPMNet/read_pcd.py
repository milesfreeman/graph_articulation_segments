import open3d as o3d 
import sys 
import numpy as np
# example usage: $ python3 ./read_pcd.py Cabinets/pts/Cabinet_01_s2_f01.pts Cabinets/seg/Cabinet_01_s2.seg
def main():

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

    pts = sys.argv[1]
    # seg = sys.argv[2]
    with open(pts, 'r') as f:
        lines = f.readlines()
    pcd = o3d.geometry.PointCloud()
    points = []
    for line in lines:
        x = line.split()
        points.append([float(x[0]), float(x[1]), float(x[2])])

    # with open(seg, 'r') as f:
    #     lines = f.readlines()
    # colores = []
    # for line in lines:
    #     colores.append(colordicc[int(line[0])])
    # pcd.colors = o3d.utility.Vector3dVector(colores)
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.RenderOption.point_size=20
    # o3d.visualization.draw_geometries([pcd])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))
    o3d.visualization.draw_geometries([mesh])
