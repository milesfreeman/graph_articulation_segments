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
from treelib import Node, Tree
from sklearn.preprocessing import normalize


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
    rotate_list = ['hinge']
    translate_list = ['slider', 'slider+']
    with open(fname, 'r') as f:
        jstring = f.read()
    units = json.loads(jstring)
    tree_dict = {}
    root = None
    for unit in units:
        parts = []
        for part in unit['parts']:
            parts.append((part['id'], part['name']))
        if unit['jointData'] is None:
            motion = ('S', None, None, None)
        elif unit['joint'] in rotate_list:
            motion = ('R', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
        elif unit['joint'] in translate_list:
            if 'rotates' in unit['jointData']['limit'].keys():
                if unit['jointData']['limit']['rotates'] and unit['jointData']['limit']['noRotationLimit']:
                    motion = ('R+T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
                else: 
                    print('fuckin up tha rotation')
                    motion = ('T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
            else:
                motion = ('T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
        else:
            # print(f"unknown joint type: {unit['joint']}")
            motion = ('S', None, None, None)
        if unit['id'] in tree_dict.keys():
            tree_dict[unit['id']].extend([unit['name'], parts, motion])
        else: 
            tree_dict[unit['id']] = [[], unit['name'], parts, motion]
        if unit['parent'] == -1: 
            root = unit['id']
        elif unit['parent'] in tree_dict.keys():
            tree_dict[unit['parent']][0].append(unit['id'])
        else:
            tree_dict[unit['parent']] = [[unit['id']]]
    
    tree = Tree()
    def add_recur(tree, node, dicc, dad):
        tree.create_node(node, node, data=dicc[node][1:], parent=dad)
        children = dicc[node][0]
        for child in children:
            add_recur(tree, child, dicc, node)
    add_recur(tree, root, tree_dict, None)
    # tree.show()
    return tree

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
    mesh_adresses = glob.glob('original-*.obj') + glob.glob('new-*.obj')
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
    
    def las_chikibeibis(tree, node):
        out = []
        for child in tree.children(node):
            out.append(child.data[1])
            if not child.is_leaf(): out.extend(las_chikibeibis(tree, child.identifier))
        return out
    
    def subtree(tree, node):
        out = []
        for child in tree.children(node):
            out.append(child.identifier)
            if not child.is_leaf(): out.extend(subtree(tree, child.identifier))
        return out

    def apply_edit(jointInfo, edit):
        if edit[0] == 'T':
            direction = edit[1][0]
            distance = edit[1][1]
            origin = jointInfo[1]
            axis = jointInfo[2]
            return (jointInfo[0], [origin[0] + distance*direction[0], origin[1] + distance*direction[1], origin[2] + distance*direction[2]], axis, jointInfo[3])
        if edit[0] == 'R':
            pts = o3d.geometry.PointCloud()
            # origin, axis direction
            pts.points = o3d.utility.Vector3dVector([jointInfo[1], jointInfo[2]])
            pts = rotate(pts, edit[1][0], edit[1][1], edit[1][2])
            origin, axis = np.asarray(pts.points)
            # axis = normalize(np.array(axis)[:, np.newaxis], axis=0).ravel().tolist()
            return (jointInfo[0], origin, axis, jointInfo[3])


        return jointInfo
    colormap = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 0.5, 0], 3: [1, 1, 0], 4: [0.5, 1, 0], 5: [0.5, 0.5, 0], 6: [0, 1, 0], 7: [0, 1, 0.5], 8: [0, 0.5, 1], 9: [0, 1, 1], 10: [0, 0.5, 0.5], 11: [0, 0, 1], 12: [1, 0, 1], 13: [0.5, 0, 0.5], 14: [0.5, 0, 1], 15: [1, 0, 0.5], 16: [0.5, 0.5, 0.5]}
    
    movement = read_json(directory + '/mobility_v2.json')
    STATIC_mesh_dicc, STATIC_pcd_dicc = align_mesh_pcd(directory)
    
    for i in range(N+1):
        mesh_dicc = copy.deepcopy(STATIC_mesh_dicc)
        pcd_dicc = copy.deepcopy(STATIC_pcd_dicc)
        axes = []
        edits = defaultdict(list)
        log = []
        for j, NODE in enumerate(movement.all_nodes()):
            unit = NODE.data
            laybel = unit[2][0]
            babies = list(itertools.chain(*las_chikibeibis(movement, NODE.identifier)))
            nametag = subtree(movement, NODE.identifier)
            # print(f"NODE: {NODE.identifier}, children: {babies}, subtree: {nametag}")
            if unit[2][0] != 'S':
                if i>0 and np.random.rand() > p: continue
                linfo = unit[2]
                if NODE.identifier in edits.keys():
                        for edit in edits[NODE.identifier]:
                            linfo = apply_edit(linfo, edit)
                if linfo[0] == 'T':
                    start = linfo[3][0]
                    end = linfo[3][1]
                    if not i: distance = 0
                    else: distance = start + np.random.rand() * (end - start)
                    update_meshes(unit[1] + babies, mesh_dicc, colormap[j%13], [trans], [[linfo[2], distance]])
                    # print(f"Translate distacne: {distance}")
                    examine_pcd(unit[1] + babies, pcd_dicc, j, laybel, [trans], [[linfo[2], distance]]) 
                    if distance != 0: 
                        for node_id in nametag:
                            edits[node_id].append(('T', [linfo[2], distance]))
                        log.append(f"T: {[NODE.identifier] + nametag}, velocity: {linfo[2]}, delta: {distance}")

                    
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
                        if not i: angle = 0
                        else: angle = start + np.random.rand() * (end - start)
                    # print(f"angle of rotation: {angle}")
                    update_meshes(unit[1] + babies, mesh_dicc, colormap[j%13], [rotate], [[linfo[1], linfo[2], angle]])
                    examine_pcd(unit[1] + babies, pcd_dicc, j,  laybel, [rotate], [[linfo[1], linfo[2], angle]])
                    if angle != 0: 
                        for node_id in nametag:
                            edits[node_id].append(('R', [linfo[1], linfo[2], angle]))
                        log.append(f"R: {[NODE.identifier] + nametag}, centroid: {linfo[1]}, velocity: {linfo[2]}, theta: {angle}")
                    
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
                    # print(f"R+T : {angle} ; {distance}")
                    update_meshes(unit[1] + babies, mesh_dicc, colormap[j%13], [rotate,trans], [[linfo[1], linfo[2], angle], [linfo[2], distance]])
                    examine_pcd(unit[1] + babies, pcd_dicc, j, laybel, [rotate,trans], [[linfo[1], linfo[2], angle], [linfo[2], distance]])
                    if angle != 0: 
                        for node_id in nametag:
                            edits[node_id].append(('R', [linfo[1], linfo[2], angle]))
                        log.append(f"R: {[NODE.identifier] + nametag}, centroid: {linfo[1]}, velocity: {linfo[2]}, theta: {angle}")
                    if distance != 0:
                        for node_id in nametag:
                            edits[node_id].append(('T', [linfo[2], distance]))
                        log.append(f"T: {[NODE.identifier] + nametag}, velocity: {linfo[2]}, delta: {distance}")

        
        mesh = []
        pcd = []
        labels = []
        for j, NODE in enumerate(movement.all_nodes()):
            pieces = list(map(lambda k: k[0], NODE.data[1]))
            for piece in pieces:
                nuage = pcd_dicc[piece]
                reiboru = [(j, NODE.data[2][0])] * len(nuage)
                pcd.extend(nuage)
                labels.extend(reiboru)
                nande = mesh_dicc[piece]
                map(lambda x: x.compute_vertex_normals(), nande)
                mesh.extend(nande)

        if visualize:
            gcloud = o3d.geometry.PointCloud()
            gcloud.points = o3d.utility.Vector3dVector(pcd) 
            print(len(mesh))
            o3d.visualization.draw_geometries(mesh  + axes, directory + '_' + ('CLOSED' if not i else str(i)))
        else: 
            save_pose(categorie, directory + '_' + ('CLOSED' if not i else str(i)), mesh, pcd, labels, log)

def examine_pcd(part_list, pcd_dicc, i, trs, transformation=None, args=None):
    for (index, _) in part_list:
        pcd = pcd_dicc[index]
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
        pcd_dicc[index] = pcd


def update_meshes(part_list, mesh_dicc, colour, transformation=None, args=None):
    for (index, name) in part_list:
        meshes = mesh_dicc[index]
        if transformation is not None:
            for i in range(len(transformation)):
                meshes = list(map(lambda x: transformation[i](x, *args[i]), meshes))
        meshes = list(map(lambda x: x.paint_uniform_color([1,0,0]), meshes))
        mesh_dicc[index] = meshes


def viewpoints(mesh, face_verts):
    def calc_norm(x):
        center = -1 * np.array(x[0])
        prev = np.array(x[1])
        faced = x[2]
        

        center /= np.linalg.norm(center)
        normal = prev/np.linalg.norm(prev)

        theta = np.arccos(np.dot(center, normal))
        phi = np.arccos(np.dot(center, -1*normal))
        # if np.isnan(normal).any(): 
        #     print(faced)
        return normal if theta < phi else -1*normal

    directions = [('R_f', 7*np.pi/4 + np.pi, 0), ('L_f', np.pi/4 + np.pi, 0), ('C_u', np.pi, np.pi/6), 
              ('R_b', 5*np.pi/4 + np.pi, 0), ('L_b', 3*np.pi/4 + np.pi, 0), ('C_d', np.pi, -np.pi/6)]    
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
        
        # pcd = []
        # for k in range(len(mesh2.face_normals)):
        #     if all(mesh2.face_normals[k] == 0):
        #         mesh2.face_colors[k] = [1,0,0]
        #         pcd.extend([i.tolist() for i in face_verts[k]])
        out.append(mesh2)
    return out



def save_pose(categorie, directory, mesh, pcd, pcd_label, moves=None):
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
    outmesh = o3d.geometry.TriangleMesh()
    outmesh.vertices = o3d.utility.Vector3dVector(v)
    outmesh.triangles = o3d.utility.Vector3iVector(f)
    outmesh.compute_triangle_normals()
    outmesh = tm.Trimesh(vertices=v, faces=f, face_normals=np.asarray(outmesh.triangle_normals))
    outmesh2 = copy.deepcopy(outmesh)
    # outmesh2.remove_degenerate_faces()
    outmesh2.fix_normals()
    # outmesh.triangle_normals = o3d.utility.Vector3dVector(nf)
    # o3d.visualization.draw_geometries([outmesh])
    # o3d.io.write_triangle_mesh(name + '.obj', outmesh)
    face_vertices = [[v[x[0]], v[x[1]], v[x[2]]] for x in f]
    os.chdir('MESH_MATLAB/' + categorie)
    Rf,Lf,Cu,Rb,Lb,Cd = viewpoints(outmesh2, face_vertices)
    if Rf == None:
        print("Model Error")
        outmesh.show()
    matlab.savemat(name + '_Lf.mat', {'vertices' : Lf.vertices,
                                   'faces' : np.asarray(Lf.faces) + 1,
                                   'normals' : Lf.face_normals})
    matlab.savemat(name + '_Rf.mat', {'vertices' : Rf.vertices,
                                   'faces' : np.asarray(Rf.faces) + 1,
                                   'normals' : Rf.face_normals})
    matlab.savemat(name + '_Cu.mat', {'vertices' : Cu.vertices,
                                   'faces' : np.asarray(Cu.faces) + 1,
                                   'normals' : Cu.face_normals})
    matlab.savemat(name + '_Lb.mat', {'vertices' : Lb.vertices,
                                   'faces' : np.asarray(Lb.faces) + 1,
                                   'normals' : Lb.face_normals})
    matlab.savemat(name + '_Rb.mat', {'vertices' : Rb.vertices,
                                   'faces' : np.asarray(Rb.faces) + 1,
                                   'normals' : Rb.face_normals})
    matlab.savemat(name + '_Cd.mat', {'vertices' : Cd.vertices,
                                   'faces' : np.asarray(Cd.faces) + 1,
                                   'normals' : Cd.face_normals})
    os.chdir('../..')
    os.chdir('PCD_LABEL/' + categorie)
    with open(name + '.txt', 'w') as f:
        f.write('# ' + directory + '\n')
        f.write('# TRANSFORMATIONS:\n')
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
    
    categorie = 'lamp'
    # os.chdir('MESH_MATLAB')
    # # os.mkdir(categorie)
    # os.chdir('../PCD_XYZ')
    # os.mkdir(categorie)
    # os.chdir('../PCD_LABEL')
    # os.mkdir(categorie)
    # os.chdir('..')
    
    with open('../category_indexes.pkl', 'rb') as f:
        models = pk.load(f)

    for model in models[categorie]:
        print(model)
        if model in ['15084', '16016', '16032', '14567', '15082', '14205', '16675', '14127', '14314'] : continue
        # if not model in ['']
        poses(categorie, model, 8, visualize=0)

main()
