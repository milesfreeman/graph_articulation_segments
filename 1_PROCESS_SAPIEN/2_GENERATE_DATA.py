import open3d as o3d
import numpy as np
import pickle as pk
import os
import sys
import glob

def gen_data(which, indices, cat, vis=False):
    # labeller = {'S' : 0, 'T' : 1, 'R' : 2}
    PCD = []
    LABEL = []
    MASK = []
    CONF = []
    extenstions = ['_CLOSED.txt'] + list(map(lambda x: '_' + str(x) + '.txt', range(1,3)))
    
    for index in indices:
        exists = glob.glob('PCD_XYZ/' + cat + '/' + index + '*.txt')
        if not bool(exists):
            # print(f"We have a fuckup: {cat}.{index}")
            print(':(')
            continue
        for ext in extenstions:
            cloud = []
            labels = np.zeros(10000, dtype=int)
            masks = np.zeros([24,10000], dtype=int)
            conf = np.zeros(24, dtype=float)
            os.chdir('PCD_XYZ/' + cat)
            with open(index + ext, 'r') as f:
                lines = f.readlines()
            for line in lines:
                x = line.split()
                cloud.append([float(x[0]), float(x[1]), float(x[2])])
            os.chdir('../..')
            os.chdir('PCD_LABEL/' + cat)
            highest = -1
            with open(index + ext, 'r') as f:
                lines = f.readlines()
            os.chdir('../..')
            for i, line in enumerate(lines):
                x = line.split()
                if int(x[0]) > highest : highest = int(x[0])
                masks[int(x[0])][i] = 1
                labels[i] = int(x[1])
            conf[ : highest + 1] = 1.0
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
                o3d.visualization.draw_geometries([oop], window_name=index + '_' + ext)

        PCD.append(cloud)
        LABEL.append(list(labels))
        MASK.append(masks)
        CONF.append(conf)
    with open(which + 'Data_' + cat + '.pkl', 'wb') as f:
        pk.dump((PCD, LABEL, MASK, CONF), f)

def yup() : return []
def main():
    
    categorie = 'storage_furniture'
    test_train_ratio = 0.75
    with open('Category_indices.pkl', 'rb') as f:
        dicc = pk.load(f)
    indices = dicc[categorie]
    split_index = int(test_train_ratio * len(indices))
    os.chdir('processed')
    gen_data('Train', indices[ : split_index], categorie)
    gen_data('Test', indices[split_index : ], categorie)

if __name__== "__main__":
    main()





    
