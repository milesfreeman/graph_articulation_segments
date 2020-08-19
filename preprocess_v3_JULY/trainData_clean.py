import open3d as o3d
import numpy as np
import pickle as pk
import os
import sys
import glob

def process_clean(models, cat, ratio=0.8, vis=False):
    labeller = {'S' : 1, 'T' : 2, 'R' : 3, 'R+T' : 4}
    PCD = []
    LABEL = []
    MASK = []
    CONF = []
    names = []
    
    for model in models:
        cloud = []
        labels = np.zeros(10000, dtype=int)
        masks = np.zeros([24,10000], dtype=int)
        conf = np.zeros(24, dtype=float)
        os.chdir('PCD_XYZ/' + cat)
        with open(model, 'r') as f:
            lines = f.readlines()
        for line in lines:
            x = line.split()
            cloud.append([float(x[0]), float(x[1]), float(x[2])])
        os.chdir('../..')
        os.chdir('PCD_LABEL/' + cat)
        highest = -1
        with open(model, 'r') as f:
            lines = f.readlines()
        os.chdir('../..')
        i=0
        for line in lines:
            x = line.split()
            if x[0] == '#' : continue
            if int(x[0]) > highest : highest = int(x[0])
            masks[int(x[0])][i] = 1
            labels[i] = labeller[x[1]]
            i+=1
        conf[ : highest + 1] = 1.0
        names.append(model[:-4])
        if vis:
            oop = o3d.geometry.PointCloud()
            oop.points = o3d.utility.Vector3dVector(cloud)
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
            colores = list(map(lambda s: colordicc[s], labels))
            oop.colors = o3d.utility.Vector3dVector(colores)
            o3d.visualization.draw_geometries([oop])
        if np.array(cloud).shape[0] != 10000:
            print('shit bitch')
            continue
        PCD.append(cloud)
        LABEL.append(list(labels))
        MASK.append(masks)
        CONF.append(conf)
    split_index = int(ratio*len(PCD))
    print(f"Train: {split_index}, Test: {len(PCD) - split_index}")
    with open('TrainData_clean_' + cat + '.pkl', 'wb') as f:
        pk.dump((PCD[:split_index], LABEL[:split_index], MASK[:split_index], CONF[:split_index]), f)
    with open('TestData_clean_' + cat + '.pkl', 'wb') as f:
        pk.dump((PCD[split_index:], LABEL[split_index:], MASK[split_index:], CONF[split_index:]), f)


def main():
    
    categorie = 'lamp'
    os.chdir('dataset/PCD_XYZ/' + categorie)
    models = glob.glob('*.txt')
    os.chdir('../..')
    process_clean(models, categorie, vis=0)
    

if __name__== "__main__":
    main()





    
