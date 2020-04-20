import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d as BN
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
import torch_geometric.data as dd
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import intersection_and_union as i_and_u
import numpy as np
import pickle as pk

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Sem_Seg(torch.nn.Module):
    def __init__(self, classes=12, k=30, aggr='max'):
        super(Sem_Seg, self).__init__()

        self.k = k
        self.out_dims = classes
        self.in_dims = 3

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), self.k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), self.k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), 
                       Dropout(0.5), 
                       MLP([256, 128]),
                       Dropout(0.5), 
                       Lin(128, self.out_dims))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

def train():
    transform = T.Compose([T.NormalizeScale(),
                           T.RandomTranslate(0.01),
                           T.RandomRotate(15, axis=0),
                           T.RandomRotate(15, axis=1),
                           T.RandomRotate(15, axis=2)])
    with open('14_CLASS.pkl', 'rb') as f:
        [X,Y] = pk.load(f)
    indices = np.array(list(range(len(X))) * 100)
    np.random.shuffle(indices)
    
    L = Correct = Nodes = 0
    for i, j in enumerate(indices):
        x = X[j]
        x = x.to(device)
        x = transform(dd.Data(pos=x)).pos
        optimizer.zero_grad()
        y = model(x)
        loss = F.nll_loss(y, Y[j].to(device))
        loss.backward()
        optimizer.step()
        L += loss.item()
        Correct += y.max(dim=1)[1].eq(Y[j].to(device)).sum().item()
        Nodes += len(x)

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(indices), L / 10,
                Correct / Nodes))
            L = Correct = Nodes = 0
    with open('weights.pkl', 'wb') as f:
        torch.save(model.state_dict(), f)

def test():
    model = Sem_Seg().to(device)
    with open('weights.pkl', 'rb') as f:
        model.load_state_dict(torch.load(f))
    with open('12_CLASS.pkl', 'rb') as f:
        [X,Y] = pk.load(f)
    loss = np.zeros([12,2])
    for i in range(25):
        x = X[i]
        x = x.to(device)
        y_pred = model(x).argmax(dim=1)
        y = Y[i].to(device)
        for j in range(len(y)):
            loss[y[j]][1] += 1
            if y_pred[j] == y[j]: loss[y[j]][0] += 1
    total = np.sum(loss, axis=0)
    np.set_printoptions(suppress=True)
    print(f"Test loss: {total[0] / total[1]}")
    weights = loss / total 
    print("Class weights")
    print(weights)
    print("Class accuracy")
    acc = loss[:,0] / loss[:,1]
    print(acc)
test()
