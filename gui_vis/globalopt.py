
'''
this is not the original prox fitting, but a modified version just for post-processing
our generated results. The input is the smplx body parameters, and the optimization 
is based on the scene sdf and the contact loss
'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import sys, os, glob
import pdb
import json
import argparse
import numpy as np
import open3d as o3d
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import smplx
from human_body_prior.tools.model_loader import load_vposer
import ChamferDistancePytorch.dist_chamfer as ext
from chamfer_python import distChamfer
from cvae import HumanCVAE, ContinousRotReprDecoder
from numpy.linalg import inv
from scipy.io import savemat
from MotionGeneration import LocalHumanDynamicsGRUNoise


BATCH_FRAME_NUM = 60
NUM_BATCHES = 5
# TODO
DCT_NUM = 5
DCT_MAT_PATH = '../Data/DCT_Basis/%d.mat' % BATCH_FRAME_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def body_params_parse(body_params_batch):
    x_body_T = body_params_batch['transl']
    x_body_R = body_params_batch['global_orient']
    x_body_beta = body_params_batch['betas']
    x_body_pose = body_params_batch['body_pose']
    x_body_lh = body_params_batch['left_hand_pose']
    x_body_rh = body_params_batch['right_hand_pose']
    x_body_CT = body_params_batch['camera_translation']
    # print(x_body_CT)
    x_body = np.concatenate([x_body_T, x_body_R,
                             x_body_beta, x_body_pose,
                             x_body_lh, x_body_rh, x_body_CT], axis=-1)
    return x_body


def get_contact_id(body_segments_folder, contact_body_parts=['L_Hand', 'R_Hand']):

    contact_verts_ids = []
    contact_faces_ids = []

    for part in contact_body_parts:
        with open(os.path.join(body_segments_folder, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
            contact_faces_ids.append(list(set(data["faces_ind"])))

    contact_verts_ids = np.concatenate(contact_verts_ids)
    contact_faces_ids = np.concatenate(contact_faces_ids)


    return contact_verts_ids, contact_faces_ids

def convert_to_6D_rot(x_batch):
    xt = x_batch[:,:3]
    xr = x_batch[:,3:6]
    xb = x_batch[:, 6:]

    xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
    xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

    return torch.cat([xt, xr_repr, xb], dim=-1)


def convert_to_3D_rot(x_batch):
    xt = x_batch[:,:3]
    xr = x_batch[:,3:9]
    xb = x_batch[:,9:]

    xr_mat = ContinousRotReprDecoder.decode(xr) # return [:,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

    return torch.cat([xt, xr_aa, xb], dim=-1)



def verts_transform(verts_batch, cam_ext_batch):
    """
    It transforms a batch of vertices by a batch of camera extrinsics
    
    Args:
      verts_batch: (batch_size, num_verts, 3)
      cam_ext_batch: (batch_size, 4, 4)
    
    Returns:
      The transformed vertices.
    """
    verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)

    verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                               cam_ext_batch.permute(0,2,1))

    verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
    
    return verts_batch_transformed



def load_dct_base():
        mtx = sio.loadmat(DCT_MAT_PATH, squeeze_me=True, struct_as_record=False)	
        mtx = mtx['D']
        mtx = mtx[:DCT_NUM]
        
        return np.array(mtx).T




class FittingOP:
    def __init__(self, fittingconfig, lossconfig, num_body):


        for key, val in fittingconfig.items():
            setattr(self, key, val)


        for key, val in lossconfig.items():
            setattr(self, key, val)

        self.batch_size = num_body
        self.vposer, _ = load_vposer(self.vposer_ckpt_path, vp_model='snapshot')
        self.body_mesh_model = smplx.create(self.human_model_path, model_type='smplx',
                                       gender='neutral', ext='npz',
                                       num_pca_comps=12,
                                       create_global_orient=True,
                                       create_body_pose=True,
                                       create_betas=True,
                                       create_left_hand_pose=True,
                                       create_right_hand_pose=True,
                                       create_expression=True,
                                       create_jaw_pose=True,
                                       create_leye_pose=True,
                                       create_reye_pose=True,
                                       create_transl=True,
                                       batch_size=self.batch_size
                                       )
        self.num_body= num_body
        self.vposer.to(self.device)
        self.body_mesh_model.to(self.device)
        print(self.scene_verts_path)
        scene_o3d = o3d.io.read_triangle_mesh(self.scene_verts_path)
        scene_verts = np.asarray(scene_o3d.vertices)
        self.s_verts_batch = torch.tensor(scene_verts, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.s_verts_batch = self.s_verts_batch.repeat(self.batch_size, 1,1)
        # self.camera_ext = extract_ext(self.camera_path)
        # scale=1.0
        self.scale = Variable(torch.tensor(1.8).to(self.device),requires_grad=True)
        self.body_rotation_rec = Variable(torch.randn(num_body,75).to(self.device), requires_grad=True)
        # print(self.scale)
        self.camera_ext = Variable(torch.randn(num_body,4,4).to(self.device),requires_grad=True)

        self.dct_mtx = load_dct_base()
        self.dct_mtx = torch.tensor(self.dct_mtx, dtype=torch.float32, device=self.device)
        self.c_dct = Variable(torch.randn(NUM_BATCHES, 23, 3, DCT_NUM).to(self.device),requires_grad=True)

        self.optimizer = optim.Adam([self.body_rotation_rec,self.scale,self.camera_ext,self.c_dct], lr=self.init_lr_h)


    def body2world(self):
        cam_transl_batch=self.body_rotation_rec[:,-3:]
        boyd2camera_list=[]
        for i in range(self.num_body):

            camera_pose = torch.eye(4)
            camera_transl = cam_transl_batch[i,:]*self.scale
            camera_pose[:3, 3] = camera_transl
            camera_pose = camera_pose.cuda()
            boyd2camera_list.append(camera_pose)

        # cam_ext_list=self.extract_ext(body_data_rotation)     
        body2camera_batch = torch.stack(boyd2camera_list, dim=0) 
        # print(cam_ext_batch.size()) 
        body2world_batch =  torch.matmul(self.camera_ext,body2camera_batch)
        return body2world_batch

    def extract_ext(self):
        lines = [line.rstrip('\n') for line in open(self.camera_path)]
        cam_ext_list=[]
        num=len(lines)

        for i in range(num):
            line = lines[i]
            items = line.split(' ')

            qvec = np.array([float(items[1]),float(items[2]),float(items[3]),float(items[4])])
            tvec = np.array([float(items[5]),float(items[6]),float(items[7])])
            romat = qvec2rotmat(qvec)
            cam_ext = np.eye(4)
            cam_ext[:3, 3] = tvec
            cam_ext[0:3, 0:3] =romat
            cam_ext = inv(cam_ext)
            cam_ext = torch.tensor(cam_ext, dtype=torch.float32).cuda()
            cam_ext_list.append(cam_ext)

        cam_ext_batch = torch.stack(cam_ext_list, dim=0) 


        return cam_ext_batch

    def cal_dctloss(self, body_joints_batch):
        """
        > For each joint, for each axis, for each batch, calculate the DCT loss
        
        Args:
          body_joints_batch: a tensor of shape (BATCH_FRAME_NUM*NUM_BATCHES, 23, 3)
        
        Returns:
          The mean of the sum of the squared error of the trajectory and the trajectory_dct
        """
        joint_ids = list(range(23)) # indices of joints in body_joints that need dct loss
        objs = {}

        for i, jid in enumerate(joint_ids):
            for j, aid in enumerate([0, 1, 2]):
                for k in range(NUM_BATCHES):
                    # trajectory = body_joints_batch[:, i, aid]
                    trajectory = body_joints_batch[BATCH_FRAME_NUM*k:BATCH_FRAME_NUM*(k+1), jid, aid]
                    trajectory_dct = torch.matmul(self.dct_mtx, torch.unsqueeze(self.c_dct[k, i, j], -1))
                    trajectory_dct = torch.squeeze(trajectory_dct)
                    error = (trajectory - trajectory_dct)**2
                    objs['DCT_%d_%d_%d' % (i, j, k)] = torch.sum( error/(error + 1.0) )
        
        return torch.mean(torch.stack(list(objs.values())))


    def cal_loss(self, body_data_rotation,idx1):
        ### reconstruction loss
        # cam_ext_list=self.extract_ext(body_data_rotation)   
        # print(self.camera_ext)  
        body2world_batch = self.body2world() 

        weights=torch.ones(body_data_rotation.size())
        weights[idx1,:]=0.0
        weights = weights.to(self.device)

        loss_rec = self.weight_loss_rec*torch.mean(torch.abs(body_data_rotation-self.body_rotation_rec)*weights)

        body_rec = convert_to_3D_rot(self.body_rotation_rec)
        vposer_pose = body_rec[:,16:48]
        loss_vposer = self.weight_loss_vposer * torch.mean(vposer_pose**2)


        diff=self.body_rotation_rec[0:-1,:]-self.body_rotation_rec[1:,:]
        loss_smoothing =  torch.mean(torch.abs(diff[0:-1,:]-diff[1:,:]))
        body_param_rec = HumanCVAE.body_params_encapsulate_batch(body_rec)

        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)

        body_param_ = {}
        for key in body_param_rec.keys():
            if key in ['body_pose_vp']:
                continue
            else:
                body_param_[key] = body_param_rec[key]

        smplx_output = self.body_mesh_model(return_verts=True, 
                                              body_pose=joint_rot_batch,
                                              **body_param_)
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch = body_verts_batch*self.scale
        body_verts_batch = verts_transform(body_verts_batch, body2world_batch)


        vid, fid = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=self.contact_part)
        body_verts_contact_batch = body_verts_batch[:, vid, :]

        dist_chamfer_contact = ext.chamferDist()
        contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch.contiguous(), 
                                                self.s_verts_batch.contiguous())
        loss_contact = self.weight_contact * torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0))
        #decrease 1.0  

        body_joints_batch = smplx_output.joints[:,0:23,:]
        body_joints_batch = verts_transform(body_joints_batch, body2world_batch)
        # print(smplx_output)
        # wordl_diff = body2world_batch[0:-1,:,:]-body2world_batch[1:,:,:]
        # wordl_diff = body_joints_batch[0:-1,:,:]-body_joints_batch[1:,:,:]
        # loss_world_smoothing = torch.mean(torch.abs(wordl_diff[0:-1,:,:]-wordl_diff[1:,:,:]))
        loss_world_smoothing= torch.mean(torch.abs(body_joints_batch[0:-1,:,:]-body_joints_batch[1:,:,:]))
        # world_diff = sum_world_diff/299.0
        # print(world_diff.size())
        # print(body_joints_batch.size())
        # print(body_joints_batch)  

        loss_dct = self.cal_dctloss(body_joints_batch)
        
        return loss_rec, loss_vposer, loss_contact, loss_smoothing, loss_world_smoothing, loss_dct


    def detect_contact(self, body_data_rotation, idx1):
        # detect contact between floor and feet

        body2world_batch = self.body2world() 

        body_rec = convert_to_3D_rot(self.body_rotation_rec)
        body_param_rec = HumanCVAE.body_params_encapsulate_batch(body_rec)

        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)

        body_param_ = {}
        for key in body_param_rec.keys():
            if key in ['body_pose_vp']:
                continue
            else:
                body_param_[key] = body_param_rec[key]

        smplx_output = self.body_mesh_model(return_verts=True, 
                                              body_pose=joint_rot_batch,
                                              **body_param_)
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch = body_verts_batch*self.scale
        body_verts_batch = verts_transform(body_verts_batch, body2world_batch)


        vid_left, _ = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=['L_Leg']) # 'L_Leg','R_Leg'
        body_verts_contact_batch_left = body_verts_batch[:, vid_left, :]

        vid_right, _ = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=['R_Leg']) # 'L_Leg','R_Leg'
        body_verts_contact_batch_right = body_verts_batch[:, vid_right, :]

        dist_chamfer_contact = ext.chamferDist()
        contact_dist_left, _ = dist_chamfer_contact(body_verts_contact_batch_left.contiguous(), 
                                                self.s_verts_batch.contiguous())
        contact_dist_right, _ = dist_chamfer_contact(body_verts_contact_batch_right.contiguous(), 
                                                self.s_verts_batch.contiguous())
        
        contact_dist_left = torch.mean(contact_dist_left, dim=1)
        contact_dist_right = torch.mean(contact_dist_right, dim=1)
        print(contact_dist_left)
        print(contact_dist_right)

        # np.savez("contact.npz", left=contact_dist_left.numpy(), right=contact_dist_right.numpy())
        # contact_dict = {"left":contact_dist_left.detach().cpu().numpy(), "right":contact_dist_right.detach().cpu().numpy()}
        # savemat("contact.mat", contact_dict)

        weight_left = contact_dist_left / (contact_dist_left + contact_dist_left)
        return weight_left.detach()

    
    def cal_loss2(self, body_data_rotation, idx1, contact_weight, left_turn):
         
        body2world_batch = self.body2world() 

        weights=torch.ones(body_data_rotation.size())
        weights[idx1,:]=0.0
        weights = weights.to(self.device)

        loss_rec = self.weight_loss_rec*torch.mean(torch.abs(body_data_rotation-self.body_rotation_rec)*weights)

        body_rec = convert_to_3D_rot(self.body_rotation_rec)

        # local smoothing
        diff_local=self.body_rotation_rec[0:-1,:]-self.body_rotation_rec[1:,:]
        loss_local_smoothing =  torch.mean(torch.abs(diff_local[0:-1,:]-diff_local[1:,:]))

        body_param_rec = HumanCVAE.body_params_encapsulate_batch(body_rec)

        joint_rot_batch = self.vposer.decode(body_param_rec['body_pose_vp'], 
                                           output_type='aa').view(self.batch_size, -1)

        body_param_ = {}
        for key in body_param_rec.keys():
            if key in ['body_pose_vp']:
                continue
            else:
                body_param_[key] = body_param_rec[key]

        smplx_output = self.body_mesh_model(return_verts=True, 
                                              body_pose=joint_rot_batch,
                                              **body_param_)
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch = body_verts_batch*self.scale
        body_verts_batch = verts_transform(body_verts_batch, body2world_batch)

        # global smoothing
        diff = body_verts_batch[0:-1,:]-body_verts_batch[1:,:]
        loss_smoothing = torch.mean(torch.abs(diff[0:-1,:]-diff[1:,:]))

        vid_left, _ = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=['L_Leg']) # 'L_Leg','R_Leg'
        body_verts_contact_batch_left = body_verts_batch[:, vid_left, :]

        vid_right, _ = get_contact_id(body_segments_folder=self.contact_id_folder,
                                contact_body_parts=['R_Leg']) # 'L_Leg','R_Leg'
        body_verts_contact_batch_right = body_verts_batch[:, vid_right, :]

        wordl_diff_left = body_verts_contact_batch_left[0:-1,:,:]-body_verts_contact_batch_left[1:,:,:]
        wordl_diff_right = body_verts_contact_batch_right[0:-1,:,:]-body_verts_contact_batch_right[1:,:,:]

        weight_right = contact_weight.clone()
        weight_left = 1 - weight_right

        weight_left[weight_left<0.5] = 0.0
        weight_right[weight_right<0.5] = 0.0

        weight_right_ = weight_right[1:]
        weight_left_ = weight_left[1:]
        weight_left_ = weight_left_.unsqueeze(1).unsqueeze(1).repeat(1, wordl_diff_left.shape[1], wordl_diff_left.shape[2])
        weight_right_ = weight_right_.unsqueeze(1).unsqueeze(1).repeat(1, wordl_diff_right.shape[1], wordl_diff_right.shape[2])
        
        loss_contact_smoothing = torch.mean(torch.abs(wordl_diff_left * weight_left_)) + torch.mean(torch.abs(wordl_diff_right * weight_right_))


        """dist_chamfer_contact = ext.chamferDist()
        if left_turn:
            contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch_left.contiguous(), 
                                                    self.s_verts_batch.contiguous()) # [300, ...]
            contact_dist = torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0), dim=1)

            loss_contact = self.weight_contact * torch.mean(contact_dist*weight_left)

        else:
            contact_dist, _ = dist_chamfer_contact(body_verts_contact_batch_right.contiguous(), 
                                                    self.s_verts_batch.contiguous()) # [300, ...]
            contact_dist = torch.mean(torch.sqrt(contact_dist+1e-4)/(torch.sqrt(contact_dist+1e-4)+1.0), dim=1)

            loss_contact = self.weight_contact * torch.mean(contact_dist*weight_right)"""

        return loss_rec, loss_local_smoothing, loss_smoothing, loss_contact_smoothing


    def init(self,body_data_rotation):
        #### we find all wrogn detection and replace the initial value with closet correct detection
        #### return: index of wrong detections

        self.body_rotation_rec.data = body_data_rotation.clone()
        self.camera_ext.data = self.extract_ext().clone()
        ### where we calculate the weights based on "consecutiveness" of the body model
        ### i.e. if the two neighboring body model are way off,one of the them should 
        ### have wrong opnepose results, and we lower the wieghts of recosntruction losss accordingl
        body_par = convert_to_3D_rot(body_data_rotation)
        vposer_pose = body_par[:,16:48]

        ## here we find all abnormal openpose detections by using vposer
        ### we set the weights of reconstruction loss to zero for those detections
        vposer_stats = torch.sum(vposer_pose**2,1)
        avg_vposer = torch.sum(vposer_stats)/300.0
        idx1 = torch.where(vposer_stats>avg_vposer*1.8)[0].cpu().numpy()

        print(idx1)

        ### given an array of 1 and 0, 1 refers to correct pose, and 0 refers to wrong pose.    
        ### we want to find the index of the closet 1 to each 0, for new initialization.
        temp = np.ones(300)
        temp[idx1]=0.0

        index_one = np.where(temp==1)[0]
        index_zero = np.where(temp==0)[0]
        w = index_one.shape[0]
        h = index_zero.shape[0]
        #
        index_one = np.tile(index_one,(h,1))
        index_zero = np.tile(index_zero,(w,1))
        index_zero = index_zero.T
        diff = np.abs(index_zero-index_one)
        pos = np.argmin(diff, axis=1)
        pos = index_one[0,pos]

        self.body_rotation_rec.data[idx1,:]=body_data_rotation[pos,:]

        return idx1

    def fitting(self, body_data, mode='local'):

        body_data_rotation = convert_to_6D_rot(body_data)

        idx1 = self.init(body_data_rotation)

        body_data_rotation=body_data_rotation.detach()

        if mode == 'local':

            for ii in range(self.num_iter):
                with torch.autograd.set_detect_anomaly(True):
                    self.optimizer.zero_grad()
                    loss_rec, loss_vposer, loss_contact, loss_smoothing, loss_world_smoothing, loss_dct = self.cal_loss(body_data_rotation,idx1)
                    if  ii < self.num_iter*0.8:
                        self.camera_ext.requires_grad=False
                        self.scale.requires_grad=True
                        self.body_rotation_rec.requires_grad=True
                        self.c_dct.requires_grad=False
                        # loss = loss_dct*10
                        loss = loss_contact*0.2 + loss_smoothing*1.0 + loss_rec
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(), loss_contact.item(), loss.item()) )
                    elif ii < self.num_iter*1:
                        self.camera_ext.requires_grad=True
                        self.scale.requires_grad=False
                        self.body_rotation_rec.requires_grad=True
                        self.c_dct.requires_grad=False
                        # loss = loss_dct*0.0001 + loss_rec*0.5 + loss_contact*0.1
                        loss = loss_rec + loss_smoothing*0.5
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(), loss_contact.item(), loss.item()) )
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    torch.cuda.empty_cache()
            
            contact_weight = self.detect_contact(body_data_rotation,idx1)

            for ii in range(int(0.4*self.num_iter)):
                with torch.autograd.set_detect_anomaly(True):
                    self.optimizer.zero_grad()
                    left_turn = bool(ii%2)
                    loss_rec, loss_local_smoothing, loss_smoothing, loss_contact_smoothing= self.cal_loss2(body_data_rotation, idx1, contact_weight, left_turn)
                    
                    self.camera_ext.requires_grad=False
                    self.scale.requires_grad=False
                    self.body_rotation_rec.requires_grad=True
                    self.c_dct.requires_grad=False
                    # loss = loss_dct*10
                    loss = loss_smoothing*1.0 + loss_local_smoothing + loss_rec + loss_contact_smoothing
                    print(self.scale)
                    # if self.verbose:
                    print('[INFO][fitting] iter={:d}, l_rec={:f}, loss_local_smoothing={:f}, loss_smoothing={:f}, loss_contact_smoothing={:f}, total_loss={:f}'.format(
                                        ii, loss_rec.item(), loss_local_smoothing.item(),
                                        loss_smoothing.item(), loss_contact_smoothing.item(), loss.item()) )
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    torch.cuda.empty_cache()
        
        elif mode == 'global':

            for ii in range(self.num_iter):
                with torch.autograd.set_detect_anomaly(True):
                    self.optimizer.zero_grad()
                    loss_rec, loss_vposer, loss_contact, loss_smoothing, loss_world_smoothing, loss_dct = self.cal_loss(body_data_rotation,idx1)
                    if  ii < self.num_iter*0.8:
                        self.camera_ext.requires_grad=False
                        self.scale.requires_grad=True
                        self.body_rotation_rec.requires_grad=True
                        self.c_dct.requires_grad=False
                        # loss = loss_dct*10
                        loss = loss_contact*0.1 + loss_smoothing*1.0 + loss_rec
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(), loss_contact.item(), loss.item()) )
                    else:
                        self.camera_ext.requires_grad=True
                        self.scale.requires_grad=False
                        self.body_rotation_rec.requires_grad=True
                        self.c_dct.requires_grad=False

                        loss = loss_rec+loss_world_smoothing*1 + loss_smoothing*0.5
                        # loss = loss_rec
                        # loss = loss_dct*0.001 + loss_rec + loss_contact*0.1
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, loss_world_smoothing={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(),loss_contact.item(), loss_world_smoothing.item(), loss.item()) )
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    torch.cuda.empty_cache()
            
        elif mode == 'dct':
            self.num_iter = 10000
            for ii in range(self.num_iter):
                with torch.autograd.set_detect_anomaly(True):
                    self.optimizer.zero_grad()
                    loss_rec, loss_vposer, loss_contact, loss_smoothing, loss_world_smoothing, loss_dct = self.cal_loss(body_data_rotation,idx1)
                    if  ii < self.num_iter*0.95:
                        self.camera_ext.requires_grad=False
                        self.scale.requires_grad=False
                        self.body_rotation_rec.requires_grad=False
                        self.c_dct.requires_grad=True

                        loss = loss_dct*10
                        # loss = loss_contact*0.1 + loss_smoothing*1.0 + loss_rec
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, loss_dct={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(), loss_contact.item(), loss_dct.item(), loss.item()) )
                    elif ii < self.num_iter*1:
                        self.camera_ext.requires_grad=False
                        self.scale.requires_grad=True
                        self.body_rotation_rec.requires_grad=True
                        self.c_dct.requires_grad=False

                        loss = loss_dct*0.0001 + loss_rec*0.5 + loss_contact*0.1
                        # loss = loss_rec + loss_smoothing*0.5
                        print(self.scale)
                        # if self.verbose:
                        print('[INFO][fitting] iter={:d}, l_rec={:f}, l_vposer={:f}, loss_smoothing={:f}, loss_contact={:f}, loss_dct={:f}, total_loss={:f}'.format(
                                            ii, loss_rec.item(), loss_vposer.item(), 
                                            loss_smoothing.item(), loss_contact.item(), loss_dct.item(), loss.item()) )
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    torch.cuda.empty_cache()

        print('[INFO][fitting] fitting finish, returning optimal value')
        body_rec =  convert_to_3D_rot(self.body_rotation_rec)
        print(self.scale)
        return body_rec, self.scale.detach().cpu().numpy().squeeze(),self.camera_ext

    def save_result(self, body_rec, scale, camera_ext, fit_path):


        if not os.path.exists(fit_path):
            os.makedirs(fit_path)
            print(fit_path)
        # print(body_rec.shape)
        body_param_list = HumanCVAE.body_params_encapsulate(body_rec, scale,camera_ext)
        # body_param_list = HumanCVAE.body_params_encapsulate(body_rec, 1.0, camera_ext)
        # print('[INFO] save results to: '+output_data_file)
        # for ii, body_param in enumerate(body_param_list):

        for i in range(len(body_param_list)):
            outputfile=fit_path+'/body_gen_'+str(i).zfill(6)+'.pkl'
            outfile = open(outputfile, 'wb')
            pickle.dump(body_param_list[i], outfile)
            outfile.close()

if __name__=='__main__':


    body_path = sys.argv[1]
    fit_path = sys.argv[2]
    mode = sys.argv[3] # mode smoothing term: 'local', 'global' or 'dct'

    sample_name = body_path.split('/')[-2]
    fittingconfig={
                # 'input_data_file': input_data_file,
                # 'output_data_file': os.path.join(fit_path,scenename+'/body_gen_{:06d}.pkl'.format(ii)),
                # 'output_data_file': '/is/ps2/yzhang/workspaces/smpl-env-gen-3d/results_baseline_postproc/realcams/'+scenename+'/body_gen_{:06d}.pkl'.format(ii),
                'scene_verts_path': '/home/miao/'+sample_name+'/meshed-poisson.ply',
                'camera_path': '/home/miao/'+sample_name+'/camerapose.txt',
                'human_model_path': './models',
                'vposer_ckpt_path': './vposer/',
                'init_lr_h': 0.005,
                'num_iter':500,
                'batch_size': 1, 
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'contact_id_folder':  './body_segments',
                'contact_part': ['L_Leg','R_Leg'],
                'verbose': False
            }


    lossconfig={
        'weight_loss_rec': 1,
        'weight_loss_vposer':0.001,
        'weight_contact': 0.1,
        'weight_collision' : 0.5
    }

    body_gen_list = sorted(glob.glob(os.path.join(body_path, 'results/*/*.pkl')))
    
    
    # print(gen_path)
#for gen_path in gen_paths:
    # body_gen_list = sorted(glob.glob(os.path.join(gen_path, '*.pkl')))
    smplifyx_list=[]
    for body_gen in body_gen_list:
        input_data_file = body_gen
        with open(input_data_file, 'rb') as f:
            body_param_input = pickle.load(f)
            # print(body_gen)
        # print(body_param_input)
        x =body_params_parse(body_param_input)
        # print(x.shape)
        smplifyx_list.append(x)

    smplifyx_data = np.vstack(smplifyx_list)
    # print(smplifyx_data.shape)
    body_gpu = torch.tensor(smplifyx_data, dtype=torch.float32).cuda()
    num_body = smplifyx_data.shape[0]
    fop = FittingOP(fittingconfig, lossconfig, num_body)

    body_rec,scale,camera_ext = fop.fitting(body_gpu, mode)
    out_path = fit_path
    # print(out_path)
    fop.save_result(body_rec, scale, camera_ext, out_path)
        # break