import pickle as pk
import numpy as np
import os
import glob

def retrieve_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    dicc = {-1: 'K'}
    for line in lines:
        x = line.split()
        if x[0] == '#': continue
        dicc[int(x[0])] = x[1]
    return dicc

dicc = {'S' : 1, 'T' : 2, 'R' : 3, 'R+T' : 4, 'K' : 5}
models = glob.glob('DATA/FINAL/chair/*.pkl')
PCD = []
LABEL = []
MASK = []
CONF = []

for model in models:
    with open(model, 'rb') as f:
        (pts, labels) = pk.load(f)
    PCD.append(np.array(pts)[:10000])
    version = model.split('_')[1]
    print(version)
    part_types = retrieve_labels('DATA/PCD_LABEL/chair/chair_' + version + '.txt')
    print(part_types)
    reiboru = []
    for pt in labels[:10000]:
        reiboru.append(dicc[part_types[pt]])
    LABEL.append(reiboru)
    mask = np.zeros([24,10000], dtype=np.dtype(int))
    maxx = -2
    for i,j in enumerate(labels[:10000]):
        if j  > maxx: maxx = j
        if j == -2: j = 23
        mask[j][i] = 1
    MASK.append(mask)
    conf = np.zeros(24)
    for i in range(maxx+1): conf[i] = 1.0
    conf[23] = 1.0
    CONF.append(conf)

train_test = 0.85
split_index = int(train_test * len(PCD))

with open('TRAINdata_chair.pkl', 'wb') as f:
    pk.dump((PCD[:split_index], LABEL[:split_index], MASK[:split_index], CONF[:split_index]), f)
with open('TESTdata_chair.pkl', 'wb') as f:
    pk.dump((PCD[split_index:], LABEL[split_index:], MASK[split_index:], CONF[split_index:]), f)
