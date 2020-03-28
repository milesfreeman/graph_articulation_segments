from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import torch as tt
import torch.nn as nn
import torch.nn.functional as F
import scipy

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class FeatureProp(nn.Module):
    def __init__(self, k, nn):
        super(FeatureProp, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = tt.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class SetAbstract(tt.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SetAbstract, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = tt.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstract(tt.nn.Module):
    def __init__(self, nn):
        super(GlobalSetAbstract, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(tt.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = tt.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PartNet(tt.nn.Module):
    def __init__(self, n_classes, n_parts):
        super(PartNet, self).__init__()
        self.d = n_classes 
        self.k = n_parts
        self.SA1 = SetAbstract(0.2, 0.2, MLP([3,64,64,131], batch_norm=0))
        self.SA2 = SetAbstract(0.25, 0.4, MLP([131, 128, 128, 259], batch_norm=0))
        self.SA3 = GlobalSetAbstract(MLP([259, 256, 512, 1024], batch_norm=0))

        self.FP1 = FeatureProp(1, MLP([1280, 256, 256]))
        self.FP2 = FeatureProp(3, MLP([384, 256, 128]))
        self.FP3 = FeatureProp(3, MLP([128, 128, 128, 128]))

        self.SemanticSeg = Seq(nn.Conv1d(128, 256, 1),
                            #    nn.BatchNorm2d()
                               nn.ReLU(),
                               nn.Conv1d(256, 256, 1),
                            #    nn.BatchNorm2d(),
                               nn.ReLU(),
                               nn.Conv1d(256, 128, 1),
                            #    nn.BatchNorm2d(),
                               nn.ReLU(),
                               nn.Conv1d(128, self.d+1, 1))

        self.InstanceSeg = Seq(nn.Conv1d(128, 256, 1),
                            #    nn.BatchNorm2d()
                               nn.ReLU(),
                               nn.Conv1d(256, 256, 1),
                            #    nn.BatchNorm2d(),
                               nn.ReLU(),
                               nn.Conv1d(256, 128, 1),
                            #    nn.BatchNorm2d(),
                               nn.ReLU(),
                               nn.Conv1d(128, self.k+1, 1),
                               nn.Softmax(dim=-1))

        self.Confidence = Seq(MLP([128, 256], batch_norm=0),
                              MLP([256, 256], batch_norm=0),
                              nn.Linear(256, self.k),
                              nn.Sigmoid())
        
    def forward(self, data):
        
        l0 = (data.type(tt.float32), data.type(tt.float32), tt.zeros(data.shape[0], dtype=tt.long))
        l1 = self.SA1(*l0)
        l2 = self.SA2(*l1)
        l3 = self.SA3(*l2)

        l2 = self.FP1(*l3, *l2)
        l1 = self.FP2(*l2, *l1)
        l0,_,_ = self.FP3(*l1, *l0)

        semantics = self.SemanticSeg(l0)
        instances = self.InstanceSeg(l0)
        masks = tt.transpose(instances[:, :, :-1], (0,2,1)) # B x K x N
        others = instances[:, :, -1]
        confidence = self.Confidence(l3[1])

        return semantics, masks, others, confidence 

def HungarianMatch(pred, gt, masks):
    batch_sz = gt.shape[0]
    n_mask = gt.shape[1]
    match_score = np.matmul(gt, np.transpose(pred, axes=[0,2,1]))
    match_score = 1 - np.divide(match_score, np.maximum(np.expand_dims(np.sum(pred, 2), 1) + np.sum(gt, 2, keedpims=1) - match_score, 1e-8))
    match_index = np.zeros((batch_sz, n_mask, 2)).astype('int32')
    for i, mask in enumerate(masks):
        row, col = scipy.optimize.linear_sum_assignment(match_score[i, :mask, :])
        match_index[i, :mask, 0] = row
        match_index[i, :mask, 1] = col 
    return match_index

def JaccardLoss(pred, gt_x, gt_conf, n_point, n_mask, dicc):
    match_index = HungarianMatch(pred, gt_x, gt_conf)
    match_index.requires_grad = 0
    dicc['match_index'] = match_index
    
    match_index_row = match_index[:, :, 0]
    index = tt.where(match_index_row >= 0)
    match_index_row = tt.cat([tt.unsqueeze(index[:, 0], -1), match_index_row.view([-1,1])], 1)
    # gt_matched = tf.gather_nd(pred, match_index_row).reshape([-1, n_mask, n_point])
    gt_matched = pred[match_index_row[:,:,0],match_index_row[:,:,1],:].reshape([-1, n_mask, n_point])
    match_index_col = match_index[:, :, 1]
    index = tt.where(match_index_col >= 0)
    match_index_col = tt.cat([tt.unsqueeze(index[:, 0]), match_index_col.view([-1,1])], dim=1)
    # pred_matched = tf.gather_nd(pred, match_index_col).reshape([-1, n_mask, n_point])
    pred_matched = pred[match_index_col[:,:,0],match_index_col[:,:,1],:].reshape([-1, n_mask, n_point])
    match_score = tt.sum(tt.mul(gt_matched, pred_matched), dim=2)
    IoU = match_score / (tt.sum(gt_matched, dim=2) + tt.sum(pred_matched, dim=2) - match_score + 1e-8) #for no zero divide
    mIoU = tt.sum(tt.mul(IoU, gt_conf), dim=1) / (tt.sum(pred_matched, dim=2) - match_score + 1e-8)
    dicc['shape_IoU'] = IoU
    return mIoU, dicc

def InstanceLoss(pred, gt_mask, gt_valid, dicc):
    k = pred.shape[1]
    n = pred.shape[2]
    mIoU, dicc = JaccardLoss(pred, gt_mask, gt_valid, n, k, dicc)
    dicc['shape_mIoU'] = mIoU
    loss = - tt.sum(mIoU)
    return loss, dicc

def ConfidenceLoss(pred_conf, gt_valid, dicc):
    batch_sz = pred_conf.shape[0]
    k = pred_conf.shape[1]
    
    IoU = dicc['shape_IoU']
    match_index = dicc['match_index']

    match_col = match_index[:, :, 1]
    index = tt.where(match_col >= 0)
    all_index = tt.cat([tt.unsqueeze(index[:, 0], -1), match_col.view([-1,1])], dim=1).view([batch_sz, k, 2])
    valid_index = tt.where(gt_valid > 0.5)
    # pred_index = tf.gather_nd(all_index, valid_index)
    pred_index = all_index[valid_index[:,0],valid_index[:,1],:]
    # valid_IoU = tf.gather_nd(IoU, valid_index)
    valid_IoU = IoU[valid_index[:,0],valid_index[:,1],:]

    # conf_y = tf.scatter_nd(pred_index, valid_index, np.array([batch_sz, n_mask]))
    conf_y = tt.zeros([batch_sz, n_mask])
    conf_y[pred_index[:, 0], pred_index[:, 1]] = valid_index
    loss_per_part = (pred_conf - conf_y) ** 2
    dicc['per_part_loss'] = loss_per_part
    
    mask_y1 = (conf_y > 0.1).astype(np.int32)
    mask_y0 = 1.0 - mask_y1 

    loss_per_shape1 = tt.sum(mask_y1 * loss_per_part, axis=-1) / tt.maximum(1e-6, tt.sum(mask_y1, axis=-1))
    loss_per_shape0 = tt.sum(mask_y0 * loss_per_part, axis=-1) / tt.maximum(1e-6, tt.sum(mask_y0, axis=-1))

    per_shape_loss = loss_per_shape0 + loss_per_shape1
    dicc['per_shape_loss'] = per_shape_loss
    loss = tt.mean(per_shape_loss)
    return loss, dicc

def OtherLoss(other_pred, gt_other):
    batch_sz = other_pred.shape[0]
    n = other_pred.shape[1]
    match_score = tt.sum(other_pred * gt_other, axis=-1)
    IoU = match_score / (tt.sum(other_pred, axis=-1) + tt.sum(gt_other, axis=-1) + 1e-8)
    loss = - tt.mean(IoU)
    return loss

def L21Norm(mask_pred, other_pred, dicc):
    n = other_pred.shape[1]
    mask = tt.cat([mask_pred, tt.unsqueeze(other_pred, dim-1)], dim=1) + 1e-6
    l21 = tt.norm(tt.norm(mask, ord=2, axis=-1), ord=1, axis=-1) / n 
    loss = tt.mean(l21)
    return loss