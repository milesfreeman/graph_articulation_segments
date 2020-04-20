import open3d as o3d 
import glob
import numpy as np 
import pickle as pk

def main():

    colormap = {
        0: [0,0,0],
        1: [1,0.5,0],
        2: [1,1,0],
        3: [0.5,1,0],
        4: [0,0,1],
        5: [1,0,0.5],
        6: [0.5,0,1],
        7: [1,0,1],
        8: [0,1,0],
        9: [0,1,1],
        10: [0,1,0.5],
        11: [0,0.5,1]
    }

    pcds = glob.glob('pts/Cabinet*.pts')
    segs = glob.glob('seg/Cabinet*.seg')

    pcds = sorted(pcds)
    segs = sorted(segs)

    segment = 0
    with open(segs[segment], 'r') as f:
        labels = f.readlines()
    POINTS = []
    LABELS = []
    MASKS = []
    VALID = []
    for pts in pcds:
        if segs[segment][4:-4] != pts[4:-8]:
            print(segs[segment][:-4])
            print( pts[:-8])
            segment += 1
            with open(segs[segment], 'r') as f:
                labels = f.readlines()
 
        with open(pts, 'r') as f:
            pcd = f.readlines()

        cloud = []
        colors = []
        for x in range(len(pcd)):
            line = pcd[x].split()
            cloud.append([float(line[0]), float(line[1]), float(line[2])])
            colors.append(int(labels[x]))
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])

        POINTS.append(cloud)
        classes =  np.unique(colors)
        putangina = np.zeros([24,2048])
        mo = np.zeros(24)
        gago = list(map(lambda x: 2 if int(x)>0 else 1, labels))
        for i,instance in enumerate(classes):
            indices = np.where(np.array(colors) == instance)
            putangina[i][indices] = 1

        mo[:len(classes)] = 1

        MASKS.append(putangina)
        VALID.append(mo)
        LABELS.append(gago)
        
    with open('RPM_data.pkl', 'wb') as f:
        pk.dump((POINTS, LABELS, MASKS, VALID), f)
main()
