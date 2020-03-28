import torch as tt
import torch.nn.functional as fn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d as BN
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
import torch_geometric.data as dd
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import intersection_and_union as i_and_u
import numpy as np
import pickle as pk
import time

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Instance_Seg(tt.nn.Module):
    # classes: num part types
    # class_weights: avg(freq) / freq(c) , for softmax cross entropy
    # k: for KNN in edge convolution
    def __init__(self, classes=12, k=30, aggr='max'):
        super(Instance_Seg, self).__init__()

        self.k = k
        self.n_classes = classes
        self.in_dims = 3 # pls add normals

        # DGCNN; we stop before final FC layer
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), self.k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), self.k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        # Semantic segmentation branch
        self.F_Sem = Seq(tt.nn.Linear(1024, 256), 
                       Dropout(0.5),
                       tt.nn.ReLU() ,
                       tt.nn.Linear(256, 128),
                       tt.nn.ReLU(),
                       Dropout(0.5), 
                       Lin(128, self.n_classes))

        # Similarity matrix branch
        self.F_sim_cnv = tt.nn.Conv1d(1024, 256, 1, 1)
        self.F_Sim = Seq(tt.nn.Linear(256, 128),
                         tt.nn.ReLU())

        # Confidence map logits
        self.F_conf_cnv = tt.nn.Conv1d(1024, 128, 1, 1)
        self.F_Conf = Seq(tt.nn.Conv2d(128, 1, 1, 1),
                          tt.nn.Sigmoid())
        
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        F = self.lin1(tt.cat([x1, x2, x3], dim=1))
        # print(list(F.shape))
        F_sem = fn.log_softmax(self.F_Sem(F), dim=1)
        F_sim = self.F_sim_cnv(F.transpose(0,1).view([1,1024,-1]))
        F_sim = self.F_Sim(F_sim.squeeze().transpose(0,1))
        # F_conf = self.F_Conf(F.transpose(0,1).view([1,1024,-1]))

        return F_sem, F_sim

# similarity matrix is distance between each point embedding
# L(i,j) = S_ij (same instance)
#          alpha * ReLU(K1 - S_ij) (same class, separate instance)
#          ReLU(K2 - S_ij) (different class)
def Similarity_Loss(F_sim, labels, K1, K2, alpha):
    F_sim = F_sim.cpu()
    gramiam = tt.matmul(F_sim, F_sim.T)
    lengths = tt.diag(gramiam).view([-1,1])
    ones = tt.ones(list(lengths.shape))
    Sim = tt.matmul(lengths.T, ones) + tt.matmul(ones.T, lengths) - 2*gramiam
    del gramiam, lengths, ones
    c1 = Sim[np.nonzero(labels[0])]
    l1 = tt.sum(c1)
    del c1 
    c2 = Sim[np.nonzero(labels[1])]
    t2 = tt.Tensor(list(c2.shape)).fill_(K1)
    l2 = alpha * tt.sum(fn.relu(t2 - c2))
    del c2, t2
    Sim = Sim.cpu()
    c3 = Sim[np.nonzero(labels[2])]
    l3 = tt.sum(fn.relu(tt.Tensor(list(c3.shape)).fill_(K2) - c3))

    return l1 + l2 + l3

def train():
    device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
    model = Instance_Seg().to(device)
    optimizer = tt.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = tt.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)



    transform = T.Compose([T.NormalizeScale(),
                            T.RandomTranslate(0.01),
                            T.RandomRotate(15, axis=0),
                            T.RandomRotate(15, axis=1),
                            T.RandomRotate(15, axis=2)])
    train_data = ['Samples/I_seg1.pkl', 'Samples/I_seg2.pkl']
    for epoch in range(20):
        start = time.time()
        with open(train_data[epoch % 2], 'rb') as f:
            [X,Y] = pk.load(f)
        end = time.time()
        print(f"Train data loaded {end-start} seconds")
        indices = np.array(list(range(len(X))))
        np.random.shuffle(indices)
        for j,i in enumerate(indices):
            optimizer.zero_grad()
            x = X[i].to(device)
            x = transform(dd.Data(pos=x)).pos
            y1,y2 = model(x)
            # labels = Y[i][1:]
            # L_sim = Similarity_Loss(y2, labels, 1,2,10)
            y_sem = Y[i][0]
            L_sem = fn.nll_loss(y1.cpu(), y_sem)
            # loss = L_sem + L_sim
            L_sem.backward()
            optimizer.step()
            del y1, y_sem
            if not (j % 10):
                print(f"[{epoch}.{j}/5.{len(X)}] Semantic Loss: {L_sem}")
        del X,Y
    for epoch in range(250):
        start = time.time()
        with open(train_data[epoch % 2], 'rb') as f:
            [X,Y] = pk.load(f)
        end = time.time()
        print(f"Train data loaded {end-start} seconds")
        indices = np.array(list(range(len(X))))
        np.random.shuffle(indices)
        for j,i in enumerate(indices):
            optimizer.zero_grad()
            x = X[i].to(device)
            x = transform(dd.Data(pos=x)).pos
            y1,y2 = model(x)
            labels = Y[i][1:]
            L_sim = 0.05* Similarity_Loss(y2, labels, 1,2,2.5)
            y_sem = Y[i][0]
            L_sem = fn.nll_loss(y1.cpu(), y_sem)
            loss = L_sem + L_sim
            loss.backward()
            optimizer.step()
            del y2, labels
            del y1, y_sem
            if not (j % 10):
                print(f"[{epoch}.{j}/20.{len(X)}]  Loss: {loss}")
        del X,Y
    with open('waites.pkl', 'wb') as f:
        tt.save(model.state_dict(), f)
