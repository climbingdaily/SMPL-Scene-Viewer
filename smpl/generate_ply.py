# -*- coding: utf-8 -*-
# @Author  : jingyi
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''
import numpy as np
import os
import pickle
import cv2 as cv
import config as cfg
from .config import SMPL_FILE

def save_ply(vertice, out_file):
    model_file = SMPL_FILE
    with open(model_file, 'rb') as f:
        smpl_model = pickle.load(f, encoding='iso-8859-1')
        face_index = smpl_model['f'].astype(np.int64)
    face_1 = np.ones((face_index.shape[0], 1))
    face_1 *= 3
    face = np.hstack((face_1, face_index)).astype(int)
    if os.path.exists(out_file):
        os.remove(out_file)
    with open(out_file, "ab") as zjy_f:
        np.savetxt(zjy_f, vertice, fmt='%f %f %f')
        np.savetxt(zjy_f, face, fmt='%d %d %d %d')
    ply_header = '''ply
format ascii 1.0
element vertex 6890
property float x
property float y
property float z
element face 13776
property list uchar int vertex_indices
end_header
    '''
    with open(out_file, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header)
        f.write(old)
