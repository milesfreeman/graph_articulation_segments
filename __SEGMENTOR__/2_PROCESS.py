import open3d as o3d 
import pickle as pk 
import numpy as np
import sys
import os

def save_result(segments, name):
    label_out = []
    os.chdir('models')
    with open('model_' + name + '.pts', 'w') as f:   
        for points, label in segments:
            label_out.append((len(points), label))
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    os.chdir('../labels')
    with open('labels_' + name + '.txt', 'w') as f:
        for i, (size, label) in enumerate(label_out):
            for _ in range(size):
                f.write(f"{i} {label}\n")
    os.chdir('..')

def run_process(category, pts, pred, names, vis=False):
    os.chdir('raw/' + category)
    mask = pred['mask']
    valid = pred['valid']
    conf = pred['conf']
    sem = pred['sem']

    n_shape = pts.shape[0]
    n_ins = mask.shape[1]
    
    colordicc = {
        0: [0,0,0], 1: [1,0,0], 2: [1,0.5,0], 3: [1,1,0],
        4: [0.5,1,0],  5: [0.5,0.5,0], 6: [0,1,0], 7: [0,1,0.5], 8: [0,0.5,1], 
        9: [0,1,1], 10: [0,0.5,0.5], 11: [0,0,1], 12: [1,0,1],
        13: [0.5,0,0.5], 14: [0.5,0,1], 15: [1,0,0.5], 16: [0.5,0.5,0.5]
    }
    
    for i in range(n_shape):
        cur_pts = pts[i, ...]
        cur_mask = mask[i, ...]
        cur_valid = valid[i, :]
        cur_conf = conf[i, :]
        cur_sem = sem[i, :]
        clouds = []
        pieces = []
        cur_conf[~cur_valid] = 0.0
        idx = np.argsort(-cur_conf)
        for j in range(n_ins):
            cur_idx = idx[j]
            if cur_valid[cur_idx]:
                part = cur_pts[np.nonzero(cur_mask[cur_idx])]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(part)
                cloud.paint_uniform_color(colordicc[j])
                clouds.append(cloud)
                pieces.append((part, cur_sem[cur_idx]))
        if vis: o3d.visualization.draw_geometries(clouds)
        save_result(pieces, names[i])


def main():
    datafile = sys.argv[1]
    with open(datafile, 'rb') as f:
        (pts, pred, names) = pk.load(f)
    
    vis = True
    category = 'storage_furniture'

    run_process(category, pts, pred, names, vis)


main()