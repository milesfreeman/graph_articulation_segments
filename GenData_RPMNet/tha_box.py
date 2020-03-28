import open3d as o3d 
import numpy as np 
import copy
import progressbar


def rotate_y(theta):
    return np.array([[np.cos(theta),  0, np.sin(theta), 0],
                     [0,              1, 0            , 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0             , 0, 0,             1]])

def rotate_x(phi):
    return np.array([[1, 0,           0,            0],
                     [0, np.cos(phi), -np.sin(phi), 0],
                     [0, np.sin(phi), np.cos(phi),  0],
                     [0, 0,           0,            1]])

def frustrum():
    box = np.zeros([590, 420, 200], dtype=np.int8)
    for z in progressbar.progressbar(range(200,400)):
        left = -1*np.tan(np.deg2rad(58.5 / 2)) * z 
        right = -1*left
        top = -1*np.tan(np.deg2rad(45.6 / 2)) * z
        bottom = -1*top 

        for x in range(int(np.rint(left)), int(np.rint(right))):
            # box[x + 295][int(np.rint(top)) + 210][z - 200] = 1
            # box[x + 295][int(np.rint(bottom)) + 210][z - 200] = 1
            
            for y in range(int(np.rint(top)), int(np.rint(bottom))):
            # box[int(np.rint(left)) + 295][y + 210][z - 200] = 1
            # box[int(np.rint(right)) + 295][y + 210][z - 200] = 1

                box[x + 295][y + 210][z - 200] = 1

    indices = np.array(np.nonzero(box))
    del box
    indices[0] -= 295
    indices[1] -= 210
    indices[2] -= 300
    indices = np.transpose(indices)
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indices)
    
    del indices
    pcd2 = copy.deepcopy(pcd)
    pcd3 = copy.deepcopy(pcd)

    pcd.transform(rotate_x(-np.pi/6))
    pcd2.transform(rotate_y(-np.pi/4))
    pcd3.transform(rotate_y(np.pi/4))
    
    print('Transforms finished')

    box1 = np.zeros([590, 420, 200], dtype=np.int8)
    box2 = np.zeros([590, 420, 200], dtype=np.int8)
    box3 = np.zeros([590, 420, 200], dtype=np.int8)

    in1 = np.transpose(np.asarray(pcd.points).astype(int))
    in2 = np.transpose(np.asarray(pcd2.points).astype(int))
    in3 = np.transpose(np.asarray(pcd3.points).astype(int))
    del pcd, pcd2, pcd3
    
    in1[0] += 295
    in1[1] += 210
    in1[2] += 300
    in2[0] += 295
    in2[1] += 210
    in2[2] += 300
    in3[0] += 295
    in3[1] += 210
    in3[2] += 300

    box1[in1] = 1
    print('box1')
    box2[in2] = 1
    print('box2')
    box3[in3] = 1
    print('box3')

    box1 += box2 + box3 
    print('Layers combined')
    points = np.transpose(np.where(box1 > 1))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    print('Rendering.......')
    o3d.visualization.draw_geometries([cloud])

frustrum()




