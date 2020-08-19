import open3d as o3d
import pickle as pk
import numpy as np
import glob
import os
import scipy.io as matlab
import copy
import json
import glob
import trimesh as tm 
import copy


colormap = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 0.5, 0], 3: [1, 1, 0], 4: [0.5, 1, 0], 5: [0.5, 0.5, 0], 6: [0, 1, 0], 7: [0, 1, 0.5], 8: [0, 0.5, 1], 9: [0, 1, 1], 10: [0, 0.5, 0.5], 11: [0, 0, 1], 12: [1, 0, 1], 13: [0.5, 0, 0.5], 14: [0.5, 0, 1], 15: [1, 0, 0.5], 16: [0.5, 0.5, 0.5]}
counter = 0
# part_labels : True -> {Arm,Head,Back,Wheel etc.} ; False -> {Static, Rotate, Translate, R+T}
#                                                                1        2        3       4

def read_json(index, part_labels=False):
    labeller = {'Chair Arm' : 'S', 'Chair Head' : 'S', 'Chair Back' : 'S', 
                'Chair Seat' : 'R+T', 'Central Support' : 'JOIN', 
                'Leg' : 'JOIN', 'Foot' : 'S', 
                'Wheel Assembly' : 'R', 'Lever' : 'R', 
                'Seat Connector' : 'S', 'Mechanical Control': 'R'}
    fname = index + '/result.json'
    with open(fname, 'r') as f:
        jstring = f.read()
    parts = json.loads(jstring)[0]['children']
    mesh_dicc = []
    _path = index + '/textured_objs/'
    for part in parts:
        
        if part['text'] in ['Controller (other)', 'handle (other)']:
            meshes = []
            for subpart in part['children']: meshes.extend([_path + x + '.obj' for x in subpart['objs']])
            mesh_dicc.append((labeller['Mechanical Control'], meshes))
        
        elif part['text'] in ['Chair Seat', 'Chair Back', 'Chair Arm', 'Chair Head']:
            meshes = []
            for subpart in part['children']:
                if 'children' not in subpart.keys(): meshes.extend([_path + x + '.obj' for x in subpart['objs']])
                else: 
                    for grandbaby in subpart['children']:
                        if 'objs' not in grandbaby.keys():
                            for greatgrand in grandbaby['children']: meshes.extend([_path + x + '.obj' for x in greatgrand['objs']])
                        else: meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
            mesh_dicc.append((labeller[part['text']], meshes))
        
        elif part['text'] == 'Chair Base':
            base = part['children'][0]
            for subpart in base['children']:
                meshes = []
                if subpart['text'] in ['Central Support', 'Seat Connector']:
                    if 'children' not in subpart.keys(): meshes.extend([_path + x + '.obj' for x in subpart['objs']])
                    else: 
                        for grandbaby in subpart['children']:
                            meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
                    mesh_dicc.append((labeller[subpart['text']], meshes))

                elif subpart['text'] == 'Mechanical Control':
                    if 'children' not in subpart.keys(): meshes.extend([_path + x + '.obj' for x in subpart['objs']])
                    else: 
                        for grandbaby in subpart['children']:
                            meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
                    mesh_dicc.append((labeller[subpart['text']], meshes))

                elif subpart['text'] in ['Controller (other)', 'handle (other)']:
                    for grandbaby in subpart['children']: meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
                    mesh_dicc.append((labeller['Mechanical Control'], meshes))
                
                elif subpart['text'] in ['Footrest Ring', 'Cylindrical (other)']:
                    for grandbaby in subpart['children']: 
                        if 'children' not in grandbaby.keys() : meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
                        else: 
                            for greatgrand in grandbaby['children']: meshes.extend([_path + x + '.obj' for x in greatgrand['objs']])
                    mesh_dicc.append((labeller['Central Support'], meshes))

                else : # subpart = Star-shaped Leg
                    for grandbaby in subpart['children']:
                        meshes = []
                        if grandbaby['text'] == 'Leg': meshes.extend([_path + x + '.obj' for x in grandbaby['objs']])
                        elif grandbaby['text'] == 'Wheel Assembly':
                            for greatgrand in grandbaby['children']: 
                                if 'objs' not in greatgrand.keys(): 
                                    for fourth in greatgrand['children']: meshes.extend([_path + x + '.obj' for x in fourth['objs']])
                                else: meshes.extend([_path + x + '.obj' for x in greatgrand['objs']])
                        mesh_dicc.append((labeller[grandbaby['text']], meshes))
    return mesh_dicc

def combine_meshes(files):
    meshes = [o3d.io.read_triangle_mesh(obj) for obj in files]
    if not len(meshes):
        print('wharrafuck')

    vertices = np.asarray(meshes[0].vertices)
    faces = np.asarray(meshes[0].triangles)
    for i in range(1,len(meshes)):
        index = len(vertices) 
        vertices = np.concatenate((vertices, np.asarray(meshes[i].vertices)), axis=0)
        faces = np.concatenate((faces, (np.asarray(meshes[i].triangles) + index)), axis=0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def process_mesh(directory):
    mesh_list = read_json(directory)
    parts = []
    base = []
    index = 1
    for part in mesh_list:
        if part[0] == 'JOIN': base.extend(part[1])
        else : parts.append((part[0], combine_meshes(part[1])))
    if len(base):
        parts.append(('S', combine_meshes(base)))
    # mesh = [part[1] for part in parts]
    # mesh = [mesh[i].paint_uniform_color(colormap[i]) for i in range(len(mesh))]
    return parts

def retrieve_save_pts(mesh, directory):
    areas = np.array(list(map(lambda x: x[1].get_surface_area(), mesh)))
    sample_sz = (areas / np.sum(areas) * 10000).astype(int)
    difference = 10000 - sum(sample_sz)
    if difference: sample_sz[-1] += difference

    pcds = list(map(lambda x: x[1][1].sample_points_uniformly(x[0]), zip(sample_sz, mesh)))
    os.chdir('PCD_XYZ/chair')
    label_str = ""
    with open('chair_' + str(counter) + '.txt', 'w') as f:
        for i in range(len(mesh)):
            # pcds[i].paint_uniform_color(colormap[i])
            for xyz in np.asarray(pcds[i].points):
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
                label_str += f"{i+1} {mesh[i][0]}\n"
    os.chdir('../..')
    os.chdir('PCD_LABEL/chair')
    with open('chair_' + str(counter) + '.txt', 'w') as f:
        f.write(f"# {directory}\n")
        f.write(label_str)
    os.chdir('../..')
    # o3d.visualization.draw_geometries(pcds)

def viewpoints(mesh, face_verts):
    # CAMERA_CUBE = tm.Trimesh()
    # CAMERA_CUBE.vertices = (np.array([[0,0,-22],[1,0,-22],[1,1,-22],[0,1,-22],[0,1,-21],[1,1,-21],[1,0,-21],[0,0,-21]]) - 0.5) / 10
    # CAMERA_CUBE.faces = [[0,2,1],[0,3,2],[2,3,4],[2,4,5],[1,2,5],[1,5,6],[0,7,4],[0,4,3],[5,4,7],[5,7,6],[0,6,7],[0,1,6]]
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
            return None
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
        # scene = tm.Scene()
        # scene.add_geometry(mesh2)
        # scene.add_geometry(CAMERA_CUBE)
        # scene.show()
        out.append(mesh2)
    return out 

def mesh_angles(parts):
    global counter
    name = 'chair_' + str(counter)
    v = None
    f = []
    nf = None
    index = 0
    for _, part in parts:
        if v is None: v = np.asarray(part.vertices)
        else: v = np.concatenate([v, np.asarray(part.vertices)])
        for face in np.asarray(part.triangles):
            f.append(list(map(lambda x: int(x) + index, face)))
        index += len(np.asarray(part.vertices))
    
    outmesh = o3d.geometry.TriangleMesh()
    outmesh.vertices = o3d.utility.Vector3dVector(v)
    outmesh.triangles = o3d.utility.Vector3iVector(f)
    outmesh.compute_triangle_normals()
    outmesh = tm.Trimesh(vertices=v, faces=f, face_normals=np.asarray(outmesh.triangle_normals))
    outmesh2 = copy.deepcopy(outmesh)
    outmesh2.remove_degenerate_faces()
    outmesh2.remove_duplicate_faces()
    outmesh2.fix_normals()

    face_vertices = [[v[x[0]], v[x[1]], v[x[2]]] for x in f]
    os.chdir('MESH_MATLAB/chair')
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
    counter += 1

with open('../ChairIndices.pkl', 'rb') as f:
    indices = pk.load(f)
    for index in indices:
       
        parts = process_mesh(index)
        retrieve_save_pts(parts, index)
        mesh_angles(parts)
           
        

