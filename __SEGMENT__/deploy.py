import numpy as np
import scipy as sp
import scipy.stats as stats
import tensorflow as tf
import importlib
import os
import sys
import pickle as pk
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from geometry_utils import *
import argparse
from progressbar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='model', help='Model name [default: model]')
parser.add_argument('--level_id', type=int, default='1', help='Level ID [default: 3]')
parser.add_argument('--num_ins', type=int, default='200', help='Max Number of Instance [default: 100]')
parser.add_argument('--pred_dir', type=str, default='predictions', help='Pred dir [default: pred]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--ckpt_dir', type=str, default='weights')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
CKPT_DIR = FLAGS.ckpt_dir

parts = ['storage_furniture/cabinet/countertop', 
         'storage_furniture/cabinet/shelf',
         'storage_furniture/cabinet/cabinet_frame' ,
         'storage_furniture/cabinet/drawer subcomponents',
         'storage_furniture/cabinet/cabinet_base subtypes',
         'storage_furniture/cabinet/cabinet_door subcomponents']

NUM_CLASSES = 3
NUM_INS = 24
GPU_INDEX = FLAGS.gpu

def predict(data):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, _, _, _, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            seg_pred, mask_pred, _, conf_pred, _ = MODEL.get_model(pc_pl, NUM_CLASSES, NUM_INS, is_training_pl)
            loader = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        ckptstate = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(CKPT_DIR, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            print("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            print("Fail to load modelfile: %s" % CKPT_DIR)

    n_shape = data.shape[0]
    n_batch = int((n_shape - 1) * 1.0 / BATCH_SIZE) + 1
    out_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=np.bool)
    out_valid = np.zeros((n_shape, NUM_INS), dtype=np.bool)
    out_conf = np.zeros((n_shape, NUM_INS), dtype=np.float32)
    out_label = np.zeros((n_shape, NUM_INS), dtype=np.uint8)
    
    bar = ProgressBar()
    batch_pts = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
    for i in bar(range(n_batch)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, n_shape)

        batch_pts[:end_idx-start_idx, ...] = data[start_idx: end_idx, ...]

        feed_dict = {pc_pl: batch_pts,
                        is_training_pl: False}

        seg_pred_val, mask_pred_val, conf_pred_val = sess.run([seg_pred, mask_pred, conf_pred], feed_dict=feed_dict)

        seg_pred_val = seg_pred_val[:end_idx-start_idx, ...]    # B x N x (C+1), #0 class is *other*
        mask_pred_val = mask_pred_val[:end_idx-start_idx, ...]  # B x K x N, no other
        conf_pred_val = conf_pred_val[:end_idx-start_idx, ...]  # B x K

        mask_pred_val[mask_pred_val < 0.5] = 0
        mask_sem_val = np.matmul(mask_pred_val, seg_pred_val)   # B x K x (C+1), #0 class is *other*
        mask_sem_idx = np.argmax(mask_sem_val, axis=-1)         # B x K

        mask_hard_val = (mask_pred_val > 0.5)
        mask_valid_val = ((np.sum(mask_hard_val, axis=-1) > 10) & (mask_sem_idx > 0))

        out_mask[start_idx: end_idx, ...] = mask_hard_val
        out_valid[start_idx: end_idx, ...] = mask_valid_val
        out_conf[start_idx: end_idx, ...] = conf_pred_val
        out_label[start_idx: end_idx, ...] = mask_sem_idx - 1
    
    return {'mask': out_mask, 'valid': out_valid, 'conf': out_conf, 'sem' : out_label}


def main():
    # data_file = sys.argv[1]
    data_file = '../TrainData_storage_furniture.pkl'
    with open(data_file, 'rb') as f:
        data = pk.load(f)
    predictions = predict(data[])
    with open('mediary.pkl', 'wb') as f:
        pk.dump((data,predictions), f)

main()
