import open3d as o3d
import pickle as pk
import sys
import numpy as np
import glob
import os
import scipy.io as matlab
import copy

def movement(model):
    files = glob.glob('mvmt/' + model + '_u*.txt')
    dicc = {}
    for fname in files:
        with open(fname, 'r') as f:
            lines = f.readlines()
        number = int(lines[1].split()[1])
        style = lines[0].split()[1]
        axis_loc = np.array(list(map(lambda x: float(x), lines[3].split()[1:])))
        axis_dir = np.array(list(map(lambda x: float(x), lines[4].split()[1:])))
        axis_dir /= np.sqrt(axis_dir[0]**2 + axis_dir[1]**2 + axis_dir[2]**2)
        if style in ['R', 'T']:
            interval = (float(lines[5].split()[1]), float(lines[5].split()[2]))
            dicc[number] = (style, axis_loc, axis_dir, interval)
        else:
            range_rot = (float(lines[6].split()[1]), float(lines[6].split()[2]))
            range_trans = (float(lines[5].split()[1]), float(lines[5].split()[2]))
            dicc[number] = (style, axis_loc, axis_dir, range_rot, range_trans)
    return dicc

def read_obj(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    objects = []
    index = 1
    faces = 0
    v = []
    f = []
    o = ''
    
    for line in lines:
        x = line.split()
        if not x : continue
        if x[0] == 'v':
            v.append([float(x[1]), float(x[2]), float(x[3])])
        if x[0] == 'f':
            f.append([int(x[1]) - index, int(x[2]) - index, int(x[3]) - index])
        if x[0] == 'g':
            o = x[1]
        if x[0] == '#':
            if v and f and o:
                index += len(v)
                faces += len(f)
                objects.append({'faces' : f,
                                'vertices' : v,
                                'name' : o})
                v = []
                f = []
                o = ''
    return objects


def rotate(mesh, centroid, normal, angle):
    
    T = np.array([[1, 0, 0, -centroid[0]],
                  [0, 1, 0, -centroid[1]],
                  [0, 0, 1, -centroid[2]],
                  [0, 0, 0, 1           ]])
    T_inv = np.array([[1, 0, 0, centroid[0]],
                  [0, 1, 0, centroid[1]],
                  [0, 0, 1, centroid[2]],
                  [0, 0, 0, 1           ]])
   
    sin_xz = normal[1] / np.sqrt(normal[1]**2 + normal[2]**2)
    cos_xz = normal[2] / np.sqrt(normal[1]**2 + normal[2]**2)

    R_x = np.array([[1, 0,      0,       0],
                    [0, cos_xz, -sin_xz, 0],
                    [0, sin_xz, cos_xz,  0],
                    [0, 0,      0,       1]])
    
    R_x_inv = np.array([[1, 0,       0,       0],
                        [0, cos_xz,  sin_xz,  0],
                        [0, -sin_xz, cos_xz,  0],
                        [0, 0,       0,       1]])
    
    sin_z = -normal[0] / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    cos_z = np.sqrt(normal[1]**2 + normal[2]**2) / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

    R_y = np.array([[cos_z,  0, sin_z, 0],
                    [0,      1, 0,     0],
                    [-sin_z, 0, cos_z, 0],
                    [0,      0, 0,     1]])
    
    R_y_inv = np.array([[cos_z,  0, -sin_z, 0],
                        [0,      1, 0,      0],
                        [sin_z,  0, cos_z,  0],
                        [0,      0, 0,      1]])

    R_z = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0, 0],
                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)),  0, 0],
                    [0,                         0,                          1, 0],
                    [0,                         0,                          0, 1]])

    for t in [T, R_x, R_y, R_z, R_y_inv, R_x_inv, T_inv]:
        mesh.transform(t)
    return mesh

def save_pose(mesh, name):
    v = None
    f = []
    nf = None
    index = 1
    for part in mesh:
        if v is None: v = np.asarray(part.vertices)
        else: v = np.concatenate([v, np.asarray(part.vertices)])
        for face in np.asarray(part.triangles):
            f.append(list(map(lambda x: int(x) + index, face)))
        if nf is None: nf = np.asarray(part.triangle_normals)
        else : nf = np.concatenate([nf, np.asarray(part.triangle_normals)])
        index += len(v)
    matlab.savemat(name + '.mat', {'vertices' : v,
                                   'faces' : f,
                                   'normals' : nf})

def save_mesh(mesh, name):
    out = []
    for part in mesh:
        out.append((np.asarray(part.vertices), np.asarray(part.triangles), np.asarray(part.triangle_normals)))
    with open(name + '.pkl', 'wb') as f:
        pk.dump(out, f)

# Saves poses as .mat file for KinectSim ; CLOSED + N poses
# name: Name of model eg. Cabinet_01
# N: number of poses to make
# p: probability that any part is moved from closed position 
def poses(name, N, meshes, axis_dicc, p=1.0, visualize=False):
    for i in range(N+1):
        mesh = []
        axes = []
        for j, part in enumerate(meshes):
            # part = rotate(part, [0,0,0], [1,0,0], 90.0)
            if j in axis_dicc.keys():
                if i>0 and np.random.rand() > p:
                    mesh.append(part)
                    continue
                info = axis_dicc[j]
                if info[0] == 'T':
                    start = info[3][0]
                    end = info[3][1]
                    if not i: distance = start
                    else: distance = start + np.random.rand() * (end - start)
                    temp = copy.deepcopy(part)
                    mesh.append(temp.translate((info[2] * distance).reshape([3,1])))
                    if visualize:
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector([axis_dicc[j][1] , axis_dicc[j][1] + axis_dicc[j][2]])
                        line.lines = o3d.utility.Vector2iVector([[0,1]])
                        line.colors = o3d.utility.Vector3dVector([[1,0,0]])
                        axes.append(line)
                elif info[0] == 'R':
                    start = info[3][0]
                    end = info[3][1]
                    if not i: angle = start
                    else: angle = start + np.random.rand() * (end - start)
                    temp = copy.deepcopy(part)
                    mesh.append(rotate(temp, info[1], info[2], angle))
                    if visualize:
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector([axis_dicc[j][1] , axis_dicc[j][1] + axis_dicc[j][2]])
                        line.lines = o3d.utility.Vector2iVector([[0,1]])
                        line.colors = o3d.utility.Vector3dVector([[0,0,1]])
                        axes.append(line)
                # Part Rotates and Translates
                else:
                    print('fuck off')
                    mesh.append(part)
            else: mesh.append(part)
        if visualize: o3d.visualization.draw_geometries(mesh + axes, name + '_' + ('CLOSED' if not i else str(i)))
        save_mesh(mesh, name + '_' + ('CLOSED' if not i else str(i)))
        save_pose(mesh, name + '_' + ('CLOSED' if not i else str(i)))

# usage: python3 ./articulate.py Cabinet_01
# argv[1] = name of model
# assumes model in 'obj' folder and motion information in directory 'mvmt'
def main():

    with open('colormap.pkl', 'rb') as f:
        colordicc = pk.load(f)

    parts = read_obj('obj/' + sys.argv[1] + '.obj')
    axis_dicc = movement(sys.argv[1])
    meshes = []
    for i,part in enumerate(parts):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(part['vertices'])
        mesh.triangles = o3d.utility.Vector3iVector(part['faces'])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(colordicc[i])
        meshes.append(mesh)
    
    os.chdir('poses')
    # Edit as seen fit
    poses(sys.argv[1], 16, meshes, axis_dicc, p=0.5, visualize=False)
