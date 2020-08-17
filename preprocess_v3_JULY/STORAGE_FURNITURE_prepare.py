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
import trimesh as tm 
import copy
import itertools

countre = 1

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
    if np.isnan(sin_xz) : sin_xz = 0
    cos_xz = normal[2] / np.sqrt(normal[1]**2 + normal[2]**2)
    if np.isnan(cos_xz) : cos_xz = 1

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
            if 'rotates' in unit['jointData']['limit'].keys():
                if unit['jointData']['limit']['rotates'] and unit['jointData']['limit']['noRotationLimit']:
                    motion = ('R+T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
                else: 
                    print('fuckin up tha rotation')
                    motion = ('T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
            else:
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
    test_PCD = []
    with open('label-10000.txt', 'r') as f:
        labels = f.readlines()
    
    for i in range(10000):
        vertex =  points[i].split()
        part_pts[int(labels[i])].append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
        test_PCD.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
    
    vT_PCD = o3d.geometry.PointCloud()
    vT_PCD.points = o3d.utility.Vector3dVector(test_PCD)
    # o3d.visualization.draw_geometries([vT_PCD])

    os.chdir('..')
    os.chdir('textured_objs')
    mesh_adresses = glob.glob('original-*.obj')
    meshes = []
    for x in mesh_adresses:
        mesh = o3d.io.read_triangle_mesh(x)
        meshes.append(mesh)
    keys = list(map(lambda x: x[0], part_pts.items()))
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
        # o3d.visualization.draw_geometries([match_pcd] + matches[key], window_name=str(key))
    # original = [int(k[9:-4]) for k in mesh_adresses]
    # print(sorted(matches.keys()))
    # exit(0)
    # assert len(original) == len(list(itertools.chain(*list(map(lambda its: its[1], matches.items())))))
    return matches, part_pts

def poses(categorie, directory, N, p=1.0, visualize=False):
    colormap = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 0.5, 0], 3: [1, 1, 0], 4: [0.5, 1, 0], 5: [0.5, 0.5, 0], 6: [0, 1, 0], 7: [0, 1, 0.5], 8: [0, 0.5, 1], 9: [0, 1, 1], 10: [0, 0.5, 0.5], 11: [0, 0, 1], 12: [1, 0, 1], 13: [0.5, 0, 0.5], 14: [0.5, 0, 1], 15: [1, 0, 0.5], 16: [0.5, 0.5, 0.5]}
    movement = read_json(directory + '/mobility_v2.json')
    mesh_dicc, pcd_dicc = align_mesh_pcd(directory)
    # STORAGE_FURNITURE...
    labeller_storage_furniture = {'cabinet_door' : 1, 'cabinet_door_surface' : 1,  # red
                'drawer' : 2, 'drawer_box' : 2, 'handle' : 2,  # orange
                'shelf' : 4, # yellow
                'cabinet_frame' : 4, 'panel_base' : 4, 'countertop' : 4, 'drawer_front' : 4, # vert
                'caster_yoke' : 5, 'wheel' : 5,
                'mirror' : 3, 'glass' : 3,
                'book' : 0 , 'other_leaf' : 0} # black
    
    for i in range(N+1):
        mesh = []
        pcd = []
        labels = []
        axes = []
        descripcion = []
        for j, unit in enumerate(movement):
            # laybel = labeller_chair[unit[0]]
            laybel = unit[2][0]
            # if unit[0] == 'handle' : o3d.visualization.draw_geometries(update_meshes(unit[1], mesh_dicc, [1,0,0]))
            if unit[2][0] != 'S':
                if i>0 and np.random.rand() > p:
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j]))
                    points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel)
                    pcd.extend(points)
                    labels.extend(reiburu)
                    descripcion.append('S')
                    continue
                linfo = unit[2]
                # print(linfo)
                if linfo[0] == 'T':
                    start = linfo[3][0]
                    end = linfo[3][1]
                    if not i: distance = start
                    else: distance = start + np.random.rand() * (end - start)
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13], [trans], [[linfo[2], distance]]))
                    points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel, [trans], [[linfo[2], distance]])
                    pcd.extend(points)
                    labels.extend(reiburu)
                    descripcion.append(f'T: {distance}')
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
                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13], [rotate], [[linfo[1], linfo[2], angle]]))
                    points, reiburu = examine_pcd(unit[1], pcd_dicc, j,  laybel, [rotate], [[linfo[1], linfo[2], angle]])
                    pcd.extend(points)
                    labels.extend(reiburu)
                    descripcion.append(f'R: {angle}')
                    if visualize:
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector([linfo[1] , np.array(linfo[1]) + np.array(linfo[2])])
                        line.lines = o3d.utility.Vector2iVector([[0,1]])
                        line.colors = o3d.utility.Vector3dVector([[0,0,1]])
                        axes.append(line)
                elif linfo[0] == 'R+T':
                    start = linfo[3][0]
                    end = linfo[3][1]
                    if not i: 
                        distance = start
                        angle = 0
                    else: 
                        distance = start + np.random.rand() * (end - start)
                        angle = np.random.ranf() * 2*np.pi

                    mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13], [rotate,trans], [[linfo[1], linfo[2], angle], [linfo[2], distance]]))
                    points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel, [rotate,trans], [[linfo[1], linfo[2], angle], [linfo[2], distance]])
                    pcd.extend(points)
                    labels.extend(reiburu)
                    descripcion.append(f'R: {angle} && T: {distance}')
            else:
                mesh.extend(update_meshes(unit[1], mesh_dicc, colormap[j%13]))
                points,reiburu = examine_pcd(unit[1], pcd_dicc, j, laybel)
                pcd.extend(points)
                labels.extend(reiburu)
                descripcion.append('S')
        
        if visualize:
            # gcloud = o3d.geometry.PointCloud()
            # gcloud.points = o3d.utility.Vector3dVector(pcd) 
            o3d.visualization.draw_geometries(mesh + axes, directory + '_' + ('CLOSED' if not i else str(i)))
        else: 
            save_pose(categorie, directory + '_' + ('CLOSED' if not i else str(i)), mesh, pcd, labels, descripcion)

def examine_pcd(part_list, pcd_dicc, i, trs, transformation=None, args=None):
    labels = []
    points = []
    for (index, _) in part_list:
        pcd = copy.deepcopy(pcd_dicc[index])
        label = [(i, trs)] * len(pcd)
        if transformation is not None:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcd)
            for k in range(len(transformation)):
                cloud = transformation[k](cloud, *args[k])
            pcd = list(np.asarray(cloud.points))
            if any(list(map(lambda k: any(np.isnan(k)), pcd))) : 
                print('fuckup found')
                exit(0)
        points.extend(pcd)
        labels.extend(label)
    return points, labels

def update_meshes(part_list, mesh_dicc, colour, transformation=None, args=None):
    out = []
    for (index, name) in part_list:
        meshes = copy.deepcopy(mesh_dicc[index])
        if transformation is not None:
            for i in range(len(transformation)):
                meshes = list(map(lambda x: transformation[i](x, *args[i]), meshes))
        meshes = list(map(lambda x: x.paint_uniform_color(colour), meshes))
        out.extend(meshes)
    return out


def viewpoints(mesh, face_verts):
    def calc_norm(x):
        center = -1 * np.array(x[0])
        prev = np.array(x[1])
        faced = x[2]
        

        center /= np.linalg.norm(center)
        normal = prev/np.linalg.norm(prev)

        theta = np.arccos(np.dot(center, normal))
        phi = np.arccos(np.dot(center, -1*normal))
        if np.isnan(normal).any(): 
            print(faced)
        return normal if theta < phi else -1*normal

    directions = [('Left', 7*np.pi/4 + np.pi, 0), ('Right', np.pi/4 + np.pi, 0), ('Centre', np.pi, np.pi/6)]
    out = []
    for view, theta, phi in directions:
        R_x = np.array([[1, 0,             0,              0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi),  0],
                        [0, 0,             0,              1]])
        R_y = np.array([[np.cos(theta),  0, np.sin(theta), 0],
                        [0,            1, 0,           0],
                        [-np.sin(theta), 0, np.cos(theta), 0],
                        [0,            0, 0,           1]])
        mesh2 = copy.deepcopy(mesh)
        mesh2.apply_transform(R_x)
        mesh2.apply_transform(R_y)
        centroids = mesh2.triangles_center
        normals = mesh2.face_normals
        mesh2.face_normals = np.array(list(map(calc_norm, zip(centroids, normals, face_verts))))
        mesh2.face_colors = [[0.7,0.7,0.7]] * len(mesh2.face_normals)
        flag = 0
        pcd = []
        for k in range(len(mesh2.face_normals)):
            if all(mesh2.face_normals[k] == 0):
                mesh2.face_colors[k] = [1,0,0]
                pcd.extend([i.tolist() for i in face_verts[k]])
                flag =1
        if flag: 
            print(pcd)
            cloud = tm.PointCloud(pcd)
            scene = tm.Scene()
            scene.add_geometry(cloud)
            scene.add_geometry(mesh2)
            # scene.show()
        # else: print('safe')
        out.append(mesh2)
    return out



def save_pose(categorie, directory, mesh, pcd, pcd_label, moves):

    def split_face(mesh, face):
        i0,i1,i2 = mesh.faces[face]
        N = len(mesh.vertices)
        v_n = mesh.triangles_center[face]
        nf_n = mesh.face_normals[face]
        mesh.vertices = np.append(mesh.vertices, [v_n], axis=0)
        mesh.faces[face] = [i0, i1, N]
        mesh.faces = np.append(mesh.faces, [[i1, i2, N],[i2, i0, N]], axis=0)
        mesh.face_normals = np.append(mesh.face_normals, [nf_n.tolist(), nf_n.tolist()], axis=0)
    global countre
    name = categorie + '_' + str(countre)
    v = None
    f = []
    nf = None
    index = 0
    for part in mesh:
        if v is None: v = np.asarray(part.vertices)
        else: v = np.concatenate([v, np.asarray(part.vertices)])
        for face in np.asarray(part.triangles):
            f.append(list(map(lambda x: int(x) + index, face)))
        # if nf is None: nf = np.asarray(part.triangle_normals)
        # else : nf = np.concatenate([nf, np.asarray(part.triangle_normals)])
        index += len(np.asarray(part.vertices))
    # print(f"Vertices : {len(v)}, Faces : {len(f)}")
    outmesh = o3d.geometry.TriangleMesh()
    outmesh.vertices = o3d.utility.Vector3dVector(v)
    outmesh.triangles = o3d.utility.Vector3iVector(f)
    outmesh.compute_triangle_normals()
    outmesh = tm.Trimesh(vertices=v, faces=f, face_normals=np.asarray(outmesh.triangle_normals))
    # TESSELATE LARGE FACES
    face_areas = outmesh.area_faces
    big_boiz = list(filter(lambda i: face_areas[i] > 2*np.median(face_areas), range(len(face_areas))))
    for face in big_boiz:
        split_face(outmesh, face)
    outmesh2 = copy.deepcopy(outmesh)
    # outmesh2.remove_degenerate_faces()
    # outmesh2.fix_normals()
    # outmesh.triangle_normals = o3d.utility.Vector3dVector(nf)
    # o3d.visualization.draw_geometries([outmesh])
    # o3d.io.write_triangle_mesh(name + '.obj', outmesh)
    # exit(0)
    face_vertices = [[v[x[0]], v[x[1]], v[x[2]]] for x in f]
    os.chdir('MESH_MATLAB/' + categorie)
    L,R,C = viewpoints(outmesh2, face_vertices)
    if L == None:
        print("Model Error")
        outmesh.show()
    # L.show()
    # return
    matlab.savemat(name + '_L.mat', {'vertices' : L.vertices,
                                   'faces' : np.asarray(L.faces) + 1,
                                   'normals' : L.face_normals})
    matlab.savemat(name + '_R.mat', {'vertices' : R.vertices,
                                   'faces' : np.asarray(R.faces) + 1,
                                   'normals' : R.face_normals})
    matlab.savemat(name + '_C.mat', {'vertices' : C.vertices,
                                   'faces' : np.asarray(C.faces) + 1,
                                   'normals' : C.face_normals})
    os.chdir('../..')
    os.chdir('PCD_LABEL/' + categorie)
    with open(name + '.txt', 'w') as f:
        f.write('# ' + directory + '\n')
        for move in moves:
            f.write('# ' + move + '\n')
        for y in pcd_label:
            f.write(f"{y[0]} {y[1]}\n")
    os.chdir('../..')
    os.chdir('PCD_XYZ/' + categorie)
    with open(name + '.txt', 'w') as f:
        for xyz in pcd:
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
    os.chdir('../..')
    countre += 1

def main():
    
    categorie = sys.argv[1]
    # os.chdir('MESH_MATLAB')
    # os.mkdir(categorie)
    # os.chdir('../PCD_XYZ')
    # os.mkdir(categorie)
    # os.chdir('../PCD_LABEL')
    # os.mkdir(categorie)
    # os.chdir('..')
    
    with open('../category_indexes.pkl', 'rb') as f:
        models = pk.load(f)

    for model in models[categorie]:
        poses(categorie, model, 8, visualize=0)

main()
