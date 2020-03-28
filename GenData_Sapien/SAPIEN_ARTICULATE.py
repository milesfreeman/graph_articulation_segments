import open3d as o3d
import pickle as pk
import sys
import numpy as np
import glob
import os
import scipy.io as matlab
import copy
import json

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
        elif unit['joint'] == 'slider':
            motion = ('T', unit['jointData']['axis']['origin'], unit['jointData']['axis']['direction'], None if unit['jointData']['limit']['noLimit'] else (unit['jointData']['limit']['a'], unit['jointData']['limit']['b']))
        else:
            print('unkwon joint type')
        units.append((parts, motion))
    return units
        
