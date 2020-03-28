import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import pickle as pk
import torch_geometric.data as dd
from collections import defaultdict

from CRF_Seg_model import CRF_Seg
from DiscriminativeLoss import DiscriminativeLoss
from NegLogLikelyLoss import NLLLoss
import torch_geometric.transforms as T

# for weighted cross entropy
with open('class_weights.pkl', 'rb') as f:
    class_weights = torch.from_numpy(pk.load(f).astype(np.float32))
batch_sz = 1
n_classes = 12
n_epoch = 100
# lets add normals/RGB
in_dims = 3
lr = 1e-2
momentum = 0.9
weight_decay = 0.0005
# make resampling work
transform = T.Compose([T.NormalizeScale(),
                           T.RandomTranslate(0.01),
                           T.RandomRotate(15, axis=0),
                           T.RandomRotate(15, axis=1),
                           T.RandomRotate(15, axis=2)])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRF_Seg(input_channels=in_dims, num_classes=n_classes, embedding_size=32)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
criterion = {
    'Embedding' : DiscriminativeLoss(1.5,0.5),
    'Semantic' : NLLLoss(class_weights)
}
criterion['Embedding'].to(device)
criterion['Semantic'].to(device)

with open('train_data.pkl', 'rb') as f:
    [Points, Classes, Instances, Sizes] = pk.load(f)

indices = np.array(list(range(40)))
inf_L1 = np.Inf
inf_L2 = np.Inf
inf_Loss = np.Inf
for epoch in range(n_epoch):
    np.random.shuffle(indices)
    scheduler.step()
    scalars = defaultdict(list)
    model.train()
    for i,j in enumerate(indices):
        x = Points[j]
        x = x.to(device)
        x = transform(dd.Data(pos=x)).pos
        y1, y2 = model(x.view([1,-1,3]))

        l1 = criterion['Semantic'](y1, Classes[j].to(device)) 
        instances = torch.from_numpy(Instances[j].astype(np.float32)).view([1,-1,Sizes[j]])
        l2 = criterion['Embedding'](y2, instances.to(device), [Sizes[j]])
        if l1 < inf_L1: inf_L1 = l1
        if l2 < inf_L2: inf_L2 = l2
        optimizer.zero_grad()
        loss = l1+l2
        loss.backward()
        if loss < inf_Loss : inf_Loss = loss
        optimizer.step()
        scalars['loss'].append(loss)
        if not ((i+1) % 10):
            print(f"[{epoch}.{i+1} / {len(indices)}] Loss = {loss} ; (Semantic={l1})(Embedding={l2})")
    print(f"------EPOCA {epoch} TERMINADA; LOSS MINIMO: {inf_Loss}------")

with open('pesos.pkl', 'wb') as f:
    torch.save(model.state_dict(), f)