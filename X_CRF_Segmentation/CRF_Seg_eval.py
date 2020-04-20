import open3d as o3d
from sklearn.cluster import MeanShift
import numpy as np
import torch
import pickle as pk
from Model import CRF_Seg

def predict(save=True):
    num_classes = 12

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRF_Seg(input_channels=3, num_classes=num_classes, embedding_size=32)
    with open('pesos.pkl', 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device)
    model.eval()

    with open('train_data.pkl', 'rb') as f:
        [Points, Classes, Instances, Sizes] = pk.load(f)
    Predictions = []
    with torch.no_grad():
        for i in range(5,10):
            x = Points[i].to(device)
            y1, y2 = model(x.view([1,-1,3]))
            y1 = y1.cpu().numpy()
            semantics = np.argmax(y1, axis=-1)
            y2 = y2.cpu().numpy()
            instances = MeanShift(0.9,n_jobs=8).fit_predict(y2.squeeze())
            Predictions.append((i, semantics, instances))
    if save:
        with open('predictions1.pkl', 'wb') as f:
            pk.dump(Predictions, f)
with open('predictions1.pkl', 'rb') as f:
    predictions = pk.load(f)
with open('train_data.pkl', 'rb') as f:
    [Points, Classes, Instances, Sizes] = pk.load(f)

cmap = {
    0: [1,0,0],
    1: [0.5,0,0],
    2: [0.5,0.5,0],
    3: [0.5,1,0],
    4: [0,0.5,0],
    5: [0,1,0],
    6: [1,1,0],
    7: [0,0,1],
    8: [0,0,0.5],
    9: [0,0.5,0.5],
    10: [0,1,0.5],
    11: [0,0.5,1],
    12: [0,1,1],
    13: [0.5,0,0.5],
    14: [1,0,1],
    15: [0.5,0.5,0.5],
    16: [1,0,0.5],
    17: [0.5,1,0.5],
    18: [0,0,0]
}
for pred in predictions:
    print(len(np.unique(pred[2])))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Points[pred[0]].numpy())
    colors = list(map(lambda x: cmap[x], pred[2]))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
