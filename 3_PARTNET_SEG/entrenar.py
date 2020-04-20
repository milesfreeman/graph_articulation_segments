import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import random
from geometry_utils import *
from progressbar import ProgressBar
from subprocess import call
import pickle as pk
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='model', help='Model name [default: model]')
parser.add_argument('--num_ins', type=int, default='24', help='Max Number of Instance [default: 100]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--seg_loss_weight', type=float, default=1.0, help='Semantic Segmentation Loss Weight [default: 1.0]')
parser.add_argument('--ins_loss_weight', type=float, default=1.0, help='Instance Segmentation Loss Weight [default: 1.0]')
parser.add_argument('--other_ins_loss_weight', type=float, default=1.0, help='Instance Segmentation for the Part *Other* Loss Weight [default: 1.0]')
parser.add_argument('--l21_norm_loss_weight', type=float, default=1.0, help='l21 Norm Loss Weight [default: 1.0]')
parser.add_argument('--conf_loss_weight', type=float, default=1.0, help='Conf Loss Weight [default: 1.0]')
parser.add_argument('--num_train_epoch_per_test', type=int, default=1, help='Number of train epochs per testing [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
MODEL = importlib.import_module(FLAGS.model)

DECAY_RATE = 0.8
DECAY_STEP = 40000.0
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 3 # binary classification (moving? y/n)
NUM_INS = FLAGS.num_ins

with open('data/TrainData_chair.pkl', 'rb') as f:
    DATA = pk.load(f)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, label_pl, gt_mask_pl, gt_valid_pl, gt_other_mask_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            seg_pred, mask_pred, other_mask_pred, conf_pred, end_points = MODEL.get_model(pc_pl, NUM_CLASSES, NUM_INS, \
                                                                               is_training_pl, bn_decay=bn_decay)
            seg_loss, end_points = MODEL.get_seg_loss(seg_pred, label_pl, end_points)
            tf.summary.scalar('seg_loss', seg_loss)

            ins_loss, end_points = MODEL.get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points)
            tf.summary.scalar('ins_loss', ins_loss)

            other_ins_loss, end_points = MODEL.get_other_ins_loss(other_mask_pred, gt_other_mask_pl, end_points)
            tf.summary.scalar('other_ins_loss', other_ins_loss)

            l21_norm_loss, end_points = MODEL.get_l21_norm(mask_pred, other_mask_pred, end_points)
            tf.summary.scalar('l21_norm_loss', l21_norm_loss)

            conf_loss, end_points = MODEL.get_conf_loss(conf_pred, gt_valid_pl, end_points)
            tf.summary.scalar('conf_loss', conf_loss)

            total_loss = FLAGS.seg_loss_weight * seg_loss + FLAGS.ins_loss_weight * ins_loss + \
                    FLAGS.other_ins_loss_weight * other_ins_loss + FLAGS.l21_norm_loss_weight * l21_norm_loss + \
                    FLAGS.conf_loss_weight * conf_loss
            tf.summary.scalar('total_loss', total_loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train', sess.graph)
        test_writer = tf.summary.FileWriter('test', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pc_pl': pc_pl,
               'label_pl': label_pl,
               'gt_mask_pl': gt_mask_pl,
               'gt_valid_pl': gt_valid_pl,
               'gt_other_mask_pl': gt_other_mask_pl,
               'is_training_pl': is_training_pl,
               'seg_pred': seg_pred,
               'mask_pred': mask_pred,
               'conf_pred': conf_pred,
               'other_mask_pred': other_mask_pred,
               'seg_loss': seg_loss,
               'ins_loss': ins_loss,
               'conf_loss': conf_loss,
               'other_ins_loss': other_ins_loss,
               'l21_norm_loss': l21_norm_loss,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'learning_rate': learning_rate,
               'bn_decay': bn_decay,
               'end_points': end_points}
        segs = []
        inss = []
        losses = []
        for epoch in range(MAX_EPOCH):
            print("**** EPOCH %d ****" % epoch)
            x,y,z=train_une_epoque(sess, ops, train_writer, epoch)
            segs.append(x)
            inss.append(10*y)
            losses.append(z)
            if (epoch+1) % FLAGS.num_train_epoch_per_test == 0:
                save_path = saver.save(sess, 'weights/' + "epoch-%d.ckpt" % epoch)
                print("Model saved in file: " + save_path)
        fig = plt.figure()
        plt.plot(list(range(len(segs))), segs, label='Class Acc.')
        plt.plot(list(range(len(segs))), inss, label='Instance mIoU')
        plt.plot(list(range(len(segs))), losses, label='Total Loss')
        plt.show()


def train_une_epoque(sess, ops, writer, epoch):

    is_training = True
    POINTS = np.array(DATA[0])
    LABELS = np.array(DATA[1])
    MASKS = np.array(DATA[2])
    VALID = np.array(DATA[3])

    n_shape = POINTS.shape[0]
    idx = np.arange(n_shape)
    np.random.shuffle(idx)

    POINTS = POINTS[idx, ...]
    LABELS = LABELS[idx, ...]
    MASKS = MASKS[idx, ...]
    VALID = VALID[idx, ...]

    POINTS = provider.jitter_point_cloud(POINTS)
    POINTS = provider.shift_point_cloud(POINTS)
    POINTS = provider.random_scale_point_cloud(POINTS)
    POINTS = provider.rotate_perturbation_point_cloud(POINTS)

    num_batch = n_shape // BATCH_SIZE
    for i in range(num_batch):
        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE

        cur_pts = POINTS[start_idx: end_idx, ...]
        cur_gt_label = LABELS[start_idx: end_idx, ...]
        cur_gt_mask = MASKS[start_idx: end_idx, ...]
        cur_gt_valid = VALID[start_idx: end_idx, ...]
        cur_gt_other_mask = np.zeros([BATCH_SIZE, NUM_POINT])

        feed_dict = {ops['pc_pl']: cur_pts,
                         ops['label_pl']: cur_gt_label,
                         ops['gt_mask_pl']: cur_gt_mask,
                         ops['gt_valid_pl']: cur_gt_valid,
                         ops['gt_other_mask_pl']: cur_gt_other_mask,
                         ops['is_training_pl']: is_training}

        summary, step, _, lr_val, bn_decay_val, seg_loss_val, ins_loss_val, other_ins_loss_val, l21_norm_loss_val, conf_loss_val, loss_val, seg_pred_val, \
                mask_pred_val, other_mask_pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['learning_rate'], ops['bn_decay'], \
                    ops['seg_loss'], ops['ins_loss'], ops['other_ins_loss'], ops['l21_norm_loss'], ops['conf_loss'], ops['loss'], \
                    ops['seg_pred'], ops['mask_pred'], ops['other_mask_pred']], feed_dict=feed_dict)

        writer.add_summary(summary, step)

        seg_pred_id = np.argmax(seg_pred_val, axis=-1)
        seg_acc = np.mean(seg_pred_id == cur_gt_label)

        print("[Epoch: %d, Batch: %d/%d, LR = %6f] Loss = %6f" % (epoch,i,num_batch,lr_val,loss_val))
        print("-----> Classification Loss: %6f (Acc: %6f) ; Instance Loss: %6f" % (seg_loss_val, seg_acc, ins_loss_val))
        return seg_acc, ins_loss_val, loss_val

train()
