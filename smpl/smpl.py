"""
This file contains the definition of the SMPL model
forward: using pose and beta calculate vertex location

function get joints: calculate joints from vertex location
"""
from __future__ import division
from numpy.core.defchararray import array

import cv2
import torch
import torch.nn as nn
import numpy as np
import os

from smpl.human_body_prior.body_model.body_model import BodyModel

try:
    import cPickle as pickle
except ImportError:
    import pickle

import smpl.config as cfg

def load_body_models(gender = 'neutral', support_dir=os.path.dirname(__file__), num_betas=10, num_dmpls=0):
    bm_fname   = os.path.join(support_dir, f'smplh/{gender}/model.npz')
    dmpl_fname = os.path.join(support_dir, f'dmpls/{gender}/model.npz')
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)#.to(comp_device)
    return bm

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
                         dim=1).view(B, 3, 3)
    return rotMat

def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)

class SMPL(nn.Module):

    def __init__(self, 
                 gender='male'):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_file: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super(SMPL, self).__init__()
        if gender == 'male':
            model_file = os.path.join(os.path.dirname(__file__), 'basicModel_m_lbs_10_207_0_v1.0.0.pkl') 
        elif gender == 'female':
            model_file = os.path.join(os.path.dirname(__file__), 'basicModel_f_lbs_10_207_0_v1.0.0.pkl') 
        elif gender == 'neutral':
            model_file = os.path.join(os.path.dirname(__file__), 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl') 
        else:
            raise ValueError('Unknown gender: %s' % gender)

        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        J_regressor = smpl_model['J_regressor'].tocoo()

        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data

        i = torch.LongTensor(np.array([row, col]))
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v,
                                                                     J_regressor_shape).to_dense())
        self.register_buffer(
            'weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer(
            'posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer(
            'v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs',
                             torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces',
                             torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(
            smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in
                     range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in
             range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = torch.from_numpy(
            np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = cfg.JOINTS_IDX
        self.requires_grad_(False)

    def forward(self, pose, beta):  # return vertices location
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,
                                        10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,
                                      207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890,
                                                                              3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(
            batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(
            batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights,
                         G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890,
                                                                               batch_size,
                                                                               4,
                                                                               4).transpose(
            0, 1)
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_full_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        return joints

    def get_leaf_joints(self, joints):
        leaf_indexes = [0, 7, 8, 12, 20, 21]
        return joints[:, leaf_indexes, :]


def get_smpl_vertices(trans: torch.Tensor,
                      poses: torch.Tensor,
                      shapes: torch.Tensor,
                      smpl: SMPL):
    vertices = smpl(poses, shapes)
    vertices += trans.unsqueeze(1)
    return vertices


def split_smpl_params(smpl_params: torch.Tensor):
    if smpl_params.size(-1) == 85:
        trans = smpl_params[..., :3].contiguous()
        poses = smpl_params[..., 3:3 + 72].contiguous()
        shapes = smpl_params[..., 3 + 72:].contiguous()
        return trans, poses, shapes
    else:
        poses = smpl_params[..., :72].contiguous()
        shapes = smpl_params[..., 72:].contiguous()
        return poses, shapes


colors = {
    # colorbline/print/copy safe:
    'light_blue': [244 / 255, 176 / 255, 132 / 255],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA


def poses_to_vertices(poses, trans=None, beta = [0] * 10, batch_size = 1024, gender='male'):
    """
    It takes in a batch of poses and returns a batch of vertices
    
    Args:
      poses: the pose parameters of the SMPL model. (N, 72, 3) or (N, 24, 3, 3)
      trans: translation of the model (N, 3)
      beta: the shape parameters of the SMPL model. (10, )
      batch_size: the number of poses to process at once. Defaults to 1024
    
    Returns:
      The vertices of the mesh.
    """

    n = len(poses)
    beta = np.array(beta)
    # if not (beta != 0).sum():
    #     beta = np.array([ 0.13718624, -0.32368565,  0.06066366,  0.22490674, -0.3380233 ,
    #                       -0.1569234,  0.32280767, -0.00115923, -0.04938826,  0.04286334])
    if beta.shape[0] != poses.shape[0] or len(beta.shape) != len(poses.shape):
        beta = beta.squeeze()[None, :].repeat(n, axis=0).astype(np.float32)

    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    smpl = SMPL(gender=gender)
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size
        ub = min(ub, n)
        cur_vertices = smpl(torch.from_numpy(poses[lb:ub]), torch.from_numpy(beta[lb:ub]))
        vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))

    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices

def smplh_to_vertices_torch(poses, trans, pose_hand, batch_size = 128, betas=torch.zeros((1, 10)), gender='male', is_cuda=True):
    assert len(poses) == len(trans)
    if is_cuda:
        is_cuda = torch.cuda.is_available()
        
    if len(betas) == 1:
        betas = betas.repeat(len(poses), 1)

    def set_var(vars):
        for i, v in enumerate(vars):  
            if not isinstance(v, torch.Tensor):  
                v = torch.from_numpy(v.astype(np.float32))
            if is_cuda and v.device.type != 'cuda':
                v = v.cuda()
            vars[i] = v
        return vars
    poses, trans, pose_hand, betas = set_var([poses, trans, pose_hand, betas])

    body_model = load_body_models(gender=gender)

    if is_cuda:
        body_model = body_model.cuda()
        
    # batch_size = 128
    n = len(poses)
    n_batch = (n + batch_size - 1) // batch_size

    vertices = []
    joints = []

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size
        ub = min(ub, n)

        body_pose_world = body_model(root_orient = poses[lb:ub, :3], 
                                     pose_body = poses[lb:ub, 3:66],
                                     trans=trans[lb:ub],
                                     pose_hand=pose_hand[lb:ub].reshape(-1, 90),
                                     betas=betas[lb:ub])
        vertices.append(body_pose_world.v)
        joints.append(body_pose_world.Jtr)

    return torch.cat(vertices), torch.cat(joints)