import pickle as pk
import sys
import numpy as np
import glob
import trimesh as tm 

def movement(model):
    files = glob.glob('mvmt/' + model + '_u*.txt')
    dicc = {}
    for fname in files:
        with open(fname, 'r') as f:
            lines = f.readlines()
        number = int(lines[1].split()[1])
        style = lines[0].split()[1]
        axis_loc = np.array(list(map(lambda x: float(x), lines[3].split()[1:])))
        axis_dir = np.array(list(map(lambda x: float(x), lines[4].split()[1:])))
        if style in ['R', 'T']:
            interval = (float(lines[5].split()[1]), float(lines[5].split()[2]))
            dicc[number] = (style, axis_loc, axis_dir, interval)
        else:
            range_rot = (float(lines[6].split()[1]), float(lines[6].split()[2]))
            range_trans = (float(lines[5].split()[1]), float(lines[5].split()[2]))
            dicc[number] = (style, axis_loc, axis_dir, range_rot, range_trans)

    return dicc

def read_obj(fname):
    with open(fname, 'r') as f:
        kwargs = tm.exchange.obj.load_obj(f, split_object=True)
    print(kwargs['faces'].shape)
    exit(0)
def main():

    colordicc = {
    0: [0,0,0],
    1: [1,0,0],
    2: [1,0.5,0],
    3: [1,1,0],
    4: [0.5,1,0],
    5: [0.5,0.5,0],
    6: [0,1,0],
    7: [0,1,0.5],
    8: [0,0.5,1],
    9: [0,1,1],
    10: [0,0.5,0.5],
    11: [0,0,1],
    12: [1,0,1],
    13: [0.5,0,0.5],
    14: [0.5,0,1],
    15: [1,0,0.5],
    16: [0.5,0.5,0.5]
}

    parts = read_obj('obj/' + sys.argv[1] + '.obj')
