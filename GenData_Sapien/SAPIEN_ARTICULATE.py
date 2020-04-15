import open3d as o3d
import pickle as pk
import sys
import numpy as np
import glob
import os
import scipy.io as matlab
import copy
import json
from collections import defaultdict
import glob

def read_obj(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    objects = []
    index = 1
    faces = 0
    v = []
    f = []
    g = ''
    
    for line in lines:
        x = line.split()
        if not x : continue
        if x[0] == 'v':
            v.append([float(x[1]), float(x[2]), float(x[3])])
        if x[0] == 'f':
            f.append([int(x[1]) - index, int(x[2]) - index, int(x[3]) - index])
        if x[0] == 'g':
            if len(v) != 0:
                index += len(v)
                faces += len(f)
                objects.append({'faces' : f,
                                'vertices' : v,
                                'name' : g})
                v = []
                f = []
            g = x[1]
    if v is not None:
        index += len(v)
        faces += len(f)
        objects.append({'faces' : f,
                        'vertices' : v,
                        'name' : g})
        v = []
        f = []

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

def trans(mesh, axis, distance):
    direction = (np.array(axis) / np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)).reshape([3,1])
    return mesh.translate(direction * distance)

def read_json(fname):
    with open(fname, 'r') as f:
        jstring = f.read()
    diccs = json.loads(jstring)
    units = []
    for unit in diccs:
        parts = []
        for part in unit['parts']:
            parts.append((part['id'], part['name']))
        if unit['jointData'] is None:
            motion = ('S', None, None, None)
        elif unit['joint'] == 'hinge':
            motion = ('R', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
        elif unit['joint'] in ['slider', 'slider+']:
            motion = ('T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
        else:
            # print(f"unkwon joint type = {unit['joint']}")
            motion = ('S', None, None, None)
        units.append((unit['name'], parts, motion))
        # print(unit['name'])
    return units

def align_mesh_pcd(directory):
    os.chdir(directory + '/point_sample')
    
    with open('pts-10000.txt', 'r') as f:
        points = f.readlines()
    
    part_pts = defaultdict(list)
    with open('label-10000.txt', 'r') as f:
        labels = f.readlines()
    
    for i in range(10000):
        vertex =  points[i].split()
        part_pts[int(labels[i])].append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
    
    os.chdir('..')
    os.chdir('textured_objs')
    mesh_adresses = glob.glob('original-*.obj')
    meshes = []
    for x in mesh_adresses:
        # print(x)
        mesh = o3d.io.read_triangle_mesh(x)
        mesh.compute_triangle_normals( )
        # print(len(mesh.vertices))
        # print(len(mesh.triangles))
        # mesh.paint_uniform_color([1,0,0])
        # o3d.visualization.draw_geometries([mesh])
        meshes.append(mesh)
    keys = list(map(lambda x: x[0], part_pts.items()))
    # print(len(meshes))
    # print(len(keys))
    # assert len(meshes) == len(keys)
    # N = len(meshes)
    similartites = np.zeros([len(meshes), len(keys)])
    part_idx = sorted(keys)
    matches = defaultdict(list)
    for i in range(len(meshes)):
        pcd = meshes[i].sample_points_uniformly()
        for j in range(len(keys)):
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(part_pts[part_idx[j]])
            similartites[i][j] = np.average(np.asarray(pcd.compute_point_cloud_distance(target)))
        match = part_idx[np.argmin(similartites[i])]
        matches[match].append(meshes[i])
    os.chdir('../..')
    # for key in keys:
    #     match_pcd = o3d.geometry.PointCloud()
    #     match_pcd.points = o3d.utility.Vector3dVector(part_pts[key])
    #     o3d.visualization.draw_geometries([match_pcd] + matches[key], window_name=str(key))

    return matches, part_pts

def poses(categorie, directory, N, p=1.0, visualize=False):
    colormap = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 0.5, 0], 3: [1, 1, 0], 4: [0.5, 1, 0], 5: [0.5, 0.5, 0], 6: [0, 1, 0], 7: [0, 1, 0.5], 8: [0, 0.5, 1], 9: [0, 1, 1], 10: [0, 0.5, 0.5], 11: [0, 0, 1], 12: [1, 0, 1], 13: [0.5, 0, 0.5], 14: [0.5, 0, 1], 15: [1, 0, 0.5], 16: [0.5, 0.5, 0.5]}
    movement = read_json(directory + '/mobility_v2.json')
    mesh_dicc, pcd_dicc = align_mesh_pcd(directory)
    # for storage furniture only...
    labeller = {'cabinet_door' : 1, 'cabinet_door_surface' : '1',
                'drawer' : 2, 'drawer_box' : 2, 'drawer_front' : 2,
                'shelf' : 3,
                'cabinet_frame' : 4, 'panel_base' : 4, 'countertop' : 4,
                'caster_yoke' : 5, 'wheel' : 5,
                'handle' : 6,
                'mirror' : 7, 'glass' : 7,
                'book' : 0 , 'other_leaf' : 0}
    # print(directory)
    for i in range(N+1):
        mesh = []
        pcd = []
        labels = []
        axes = []
        for j, unit in enumerate(movement):
            laybel = labeller[unit[0]]
            if unit[2][0] != 'S':
                if i>0 and np.random.rand() > p:
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j]))
                    points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel)
                    pcd.extend(points)
                    labels.extend(reiburu)
                    continue
                linfo = unit[2]
                # print(linfo)
                if linfo[0] == 'T':
                    start = linfo[3][0]
                    end = linfo[3][1]
                    if not i: distance = start
                    else: distance = start + np.random.rand() * (end - start)
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13], trans, [linfo[2], distance]))
                    points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel, trans, [linfo[2], distance])
                    pcd.extend(points)
                    labels.extend(reiburu)
                    if visualize:
                            line = o3d.geometry.LineSet()
                            line.points = o3d.utility.Vector3dVector([linfo[1] , np.array(linfo[1]) + np.array(linfo[2])])
                            line.lines = o3d.utility.Vector2iVector([[0,1]])
                            line.colors = o3d.utility.Vector3dVector([[1,0,0]])
                            axes.append(line)
                elif linfo[0] == 'R':
                    if linfo[3] == None:
                        angle = 0 if not i else np.random.randint(0,360)
                    else:
                        start = linfo[3][0]
                        end = linfo[3][1]
                        if not i: angle = start
                        else: angle = start + np.random.rand() * (end - start)
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13], rotate, [linfo[1], linfo[2], angle]))
                    points, reiburu = examine_pcd(unit[1], pcd_dicc, j,  laybel, rotate, [linfo[1], linfo[2], angle])
                    pcd.extend(points)
                    labels.extend(reiburu)
                    if visualize:
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector([linfo[1] , np.array(linfo[1]) + np.array(linfo[2])])
                        line.lines = o3d.utility.Vector2iVector([[0,1]])
                        line.colors = o3d.utility.Vector3dVector([[0,0,1]])
                        axes.append(line)
            else:
                mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13]))
                points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel)
                pcd.extend(points)
                labels.extend(reiburu)
        
        if visualize:
            gcloud = o3d.geometry.PointCloud()
            gcloud.points = o3d.utility.Vector3dVector(pcd) 
            o3d.visualization.draw_geometries(mesh + axes + [gcloud], directory + '_' + ('CLOSED' if not i else str(i)))
        else: 
            save_pose(categorie, directory + '_' + ('CLOSED' if not i else str(i)), mesh, pcd, labels)

def examine_pcd(part_list, pcd_dicc, i, trs, transformation=None, args=None):
    labels = []
    points = []
    for (index, _) in part_list:
        pcd = copy.deepcopy(pcd_dicc[index])
        label = [(i, trs)] * len(pcd)
        if transformation is not None:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcd)
            cloud = transformation(cloud, *args)
            pcd = list(np.asarray(cloud.points))
        points.extend(pcd)
        labels.extend(label)
    return points, labels

def update_meshes(part_list, mesh_dicc, colour, transformation=None, args=None):
    out = []
    for (index, name) in part_list:
        meshes = copy.deepcopy(mesh_dicc[index])
        if transformation is not None:
            meshes = list(map(lambda x: transformation(x, *args), meshes))
        meshes = list(map(lambda x: x.paint_uniform_color(colour), meshes))
        out.extend(meshes)
    return out

def save_pose(categorie, name, mesh, pcd, pcd_label):
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
        index += len(np.asarray(part.vertices))
    # print(f"Vertices : {len(v)}, Faces : {len(f)}")
    os.chdir('MESH_MATLAB/' + categorie)
    matlab.savemat(name + '.mat', {'vertices' : v,
                                   'faces' : f,
                                   'normals' : nf})
    os.chdir('../..')
    os.chdir('PCD_LABEL/' + categorie)
    with open(name + '.txt', 'w') as f:
        for y in pcd_label:
            f.write(f"{y[0]} {y[1]}\n")
    os.chdir('../..')
    os.chdir('PCD_XYZ/' + categorie)
    with open(name + '.txt', 'w') as f:
        for xyz in pcd:
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
    os.chdir('../..')
def yup(): return []
def main():
    
    categorie = sys.argv[1]
    # os.chdir('MESH_MATLAB')
    # os.mkdir(categorie)
    # os.chdir('../PCD_XYZ')
    # os.mkdir(categorie)
    # os.chdir('../PCD_LABEL')
    # os.mkdir(categorie)
    # os.chdir('..')

    
    with open('../Category_indices.pkl', 'rb') as f:
        models = pk.load(f)
    # with open('plox.txt', 'r') as f:
    #     ja_feitos = f.readlines()

    for model in models[categorie]:
        # if model in ja_feitos: continue
        # if model in ['41434', '40069', '41045'] : continue
        poses(categorie, model, 16, visualize=0)

main()