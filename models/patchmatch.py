import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import cv2
import numpy as np
import math
from utils import *

from torch.autograd import Function
from torch.onnx.symbolic_helper import parse_args

#from test_trt import *

# engine = get_engine("./checkpoints/removed.onnx", p3_model_path)
# inputs, outputs_, bindings, stream = common.allocate_buffers(engine)
# context = engine.create_execution_context()

class DepthInitialization(nn.Module):
    def __init__(self, patchmatch_num_sample = 1):
        super(DepthInitialization, self).__init__()
        self.patchmatch_num_sample = patchmatch_num_sample

    
    def forward(self, random_initialization, min_depth,max_depth,hw, depth_interval_scale=None,
                depth=None):
        # min_depth = ddhw[0]
        # max_depth = ddhw[1]
        height = int(hw[0])
        width = int(hw[1])

        batch_size = min_depth.size()[0]

        if random_initialization:
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            inverse_min_depth = 1.0 / min_depth
            inverse_max_depth = 1.0 / max_depth
            patchmatch_num_sample = 48 
            # [B,Ndepth,H,W]
            # depth_sample = torch.ones([1,48,150,200])
            # depth_sample = torch.load("./depth_sample.pt")
            depth_sample = torch.rand((batch_size, patchmatch_num_sample, height, width), device=0) + \
                            torch.arange(0, patchmatch_num_sample, 1, device=0).view(1, patchmatch_num_sample, 1, 1)
            # depth_sample = torch.arange(0, patchmatch_num_sample*height*width, 1, device=0).view(1, patchmatch_num_sample, height, width)

            # torch.save(depth_sample,"./depth_sample.pt")

            depth_sample = inverse_max_depth.view(batch_size,1,1,1) + depth_sample / patchmatch_num_sample * \
                                    (inverse_min_depth.view(batch_size,1,1,1) - inverse_max_depth.view(batch_size,1,1,1))
            
            depth_sample = 1.0 / depth_sample
           
            return depth_sample
            
        else:
            # other Patchmatch, local perturbation is performed based on previous result
            # uniform samples in an inversed depth range
            if self.patchmatch_num_sample == 1:
                return depth.detach()
            else:
                inverse_min_depth = 1.0 / min_depth
                inverse_max_depth = 1.0 / max_depth
                
                depth_sample = torch.arange(-self.patchmatch_num_sample//2, self.patchmatch_num_sample//2, 1,
                                    device=0).view(1, self.patchmatch_num_sample, 1, 1).repeat(batch_size,
                                    1, height, width).float()
                inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_interval_scale
                inverse_depth_interval = inverse_depth_interval.view(batch_size,1,1,1)
                
                depth_sample = 1.0 / depth.detach() + inverse_depth_interval * depth_sample
                
                depth_clamped = []
                #del depth
                for k in range(batch_size):
                    depth_clamped.append(torch.clamp(depth_sample[k], min=inverse_max_depth[k], max=inverse_min_depth[k]).unsqueeze(0))
                    #depth_clamped.append(torch.clamp(depth_sample[k], min=inverse_max_depth.item(), max=inverse_min_depth.item()).unsqueeze(0))
                depth_sample = 1.0 / torch.cat(depth_clamped,dim=0)
                #del depth_clamped
                
                return depth_sample
                

class Propagation(nn.Module):
    def __init__(self, neighbors = 16,propagation_out_range=2):
        super(Propagation, self).__init__()
        self.neighbors = neighbors
        self.propagation_out_range = propagation_out_range - 1
        #self.conv_2d_feature = nn.Conv2d(1,self.neighbors,1,1,0)
        self.conv_2d_feature = ConvBn(1, self.neighbors, 1, 1, 0)
        nn.init.constant_(self.conv_2d_feature.conv.weight,1)
        # for param in self.conv_2d_feature.parameters():
        #     param.requires_grad = False
    
    def forward(self, depth_sample, offset):
        # [B,D,H,W]
        num_depth = depth_sample.size()[1]
        batch, neighbors, height, width = offset.size()
        neighbors = int(neighbors / 2)
        propogate_depth = depth_sample.new_empty(batch, num_depth + self.neighbors, height, width)
        propogate_depth[:,0:num_depth,:,:] = depth_sample

        ###--------jiangpf----------

        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d(int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range) * 2 + 1) #  + 1

            ref_feature_indx = padding(depth_sample[i])
            ref_feature_indx1 = padding1(depth_sample[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[num_depth // 2, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s + neighbors, :, :] * ref_feature_indx[num_depth // 2, indy:height + indy,
                                                                indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1) / 2
            ref_feature_list.append(ref_feature_sum)
        ref_feature = torch.stack(ref_feature_list).unsqueeze(1)
        
        #propogate_depth_sample = ref_feature
        propogate_depth_sample = self.conv_2d_feature(ref_feature)
        propogate_depth_sample = ref_feature + propogate_depth_sample
        #propogate_depth_sample = depth_sample[:,num_depth // 2,:,:].unsqueeze(1) + propogate_depth_sample

        propogate_depth[:,num_depth:,:,:] = propogate_depth_sample
        mask_naget = propogate_depth<=0
        propogate_depth[mask_naget] = 1
        # del propogate_depth_sample
        
        # sort
        # propogate_depth, _ = torch.sort(propogate_depth, dim=1)

        return propogate_depth
        
class Depth_weight(nn.Module):
    def __init__(self,propagation_range):
        super(Depth_weight, self).__init__()
        self.propagation_out_range = propagation_range - 1
    def forward(self,depth_sample, depth_min, depth_max,offset, patchmatch_interval_scale, evaluate_neighbors):
        # grid: position of sampling points in adaptive spatial cost aggregation
        neighbors = evaluate_neighbors
        batch, num_depth, height, width = depth_sample.size()
        _, cordin, hight, weight = offset.size()
        # normalization
        x = 1.0 / depth_sample
        # del depth_sample
        inverse_depth_min = 1.0 / depth_min
        inverse_depth_max = 1.0 / depth_max
        x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min.view(batch, 1, 1, 1) \
                                                            - inverse_depth_max.view(batch, 1, 1, 1))

        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d(int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range) * 2)
            # padding = nn.ConstantPad2d(int(pad_num - 1), 0)
            ref_feature_indx = padding(x[i])
            ref_feature_indx1 = padding1(x[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[:, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s + neighbors, :, :] * ref_feature_indx[:, indy:height + indy,
                                                                indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1) / 2

            ref_feature_list.append(ref_feature_sum)
        x1_ = torch.stack(ref_feature_list)

        x1_ = torch.abs(x1_ - x) / patchmatch_interval_scale
        x1_ = torch.clamp(x1_, min=0, max=4)
        x1_ = (-x1_ + 2) * 2
        output = nn.Sigmoid()
        x1_ = output(x1_)

        return x1_

class Evaluation(nn.Module):
    def __init__(self,  G=8, stage=3, evaluate_neighbors=9, iterations=2, propagation_out_range =2):
        super(Evaluation, self).__init__()
        
        self.iterations = iterations
        self.propagation_out_range = propagation_out_range - 1
        
        self.G = G
        self.stage = stage
        #if self.stage == 3:
        self.pixel_wise_net = PixelwiseNet(self.G)

        if self.stage == 0:
            self.intr_mat = torch.tensor([[2892.33008,0,823.20398],[0,2883.16992,619.07001],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[3.4574e-4, 0, -0.28461620],[0, 3.4684e-4, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 1:
            self.intr_mat = torch.tensor([[1446.16504,0,411.60199],[0,1441.58496,309.53500],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[6.9148e-4, 0, -0.28461620],[0, 6.9368e-4, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 2:
            self.intr_mat = torch.tensor([[723.08252,0,205.80099],[0,720.79248,154.76750],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[1.38297e-3, 0, -0.28461620],[0, 1.38736e-3, -0.21471854],[0, 0, 1]],
                                        device="cuda")
        elif self.stage == 3:
            self.intr_mat = torch.tensor([[361.54126,0,102.90050],[0,360.39624,77.38375],[0,0,1]],device="cuda")
            self.intr_mat_inv = torch.tensor([[2.76594e-3,0,-0.28461620],[0,2.77472e-3,-0.21471854],[0,0,1]],device="cuda")
        
        self.similarity_net = SimilarityNet(self.G, evaluate_neighbors, self.stage, propagation_out_range=propagation_out_range)
        self.similarity_net1 = SimilarityNet(self.G, evaluate_neighbors, self.stage, propagation_out_range=propagation_out_range)

    def bilinear_interpolate_torch(self, im, y, x):
        '''
           im : B,C,H,W
           y : 1,numPoints -- pixel location y float
           x : 1,numPOints -- pixel location y float
        '''
        batch, _, _, _ = im.size()
        x0 = torch.floor(x).type(torch.cuda.LongTensor)
        x1 = x0 + 1

        y0 = torch.floor(y).type(torch.cuda.LongTensor)
        y1 = y0 + 1

        wa = (x1.type(torch.cuda.FloatTensor) - x) * (y1.type(torch.cuda.FloatTensor) - y)
        wb = (x1.type(torch.cuda.FloatTensor) - x) * (y - y0.type(torch.cuda.FloatTensor))
        wc = (x - x0.type(torch.cuda.FloatTensor)) * (y1.type(torch.cuda.FloatTensor) - y)
        wd = (x - x0.type(torch.cuda.FloatTensor)) * (y - y0.type(torch.cuda.FloatTensor))
        # Instead of clamp
        n1 = x1 / im.shape[3]
        n2 = y1 / im.shape[2]
        x1 = x1 - torch.floor(n1).int()
        y1 = y1 - torch.floor(n2).int()
        Ia = []
        Ib = []
        Ic = []
        Id = []
        for i in range(batch):
            Ia.append(im[i:i + 1, :, y0[i], x0[i]])
            Ib.append(im[i:i + 1, :, y1[i], x0[i]])
            Ic.append(im[i:i + 1, :, y0[i], x1[i]])
            Id.append(im[i:i + 1, :, y1[i], x1[i]])
        Ia = torch.cat(Ia, dim=0)
        Ib = torch.cat(Ib, dim=0)
        Ic = torch.cat(Ic, dim=0)
        Id = torch.cat(Id, dim=0)
        wa = wa.unsqueeze(1)
        wb = wb.unsqueeze(1)
        wc = wc.unsqueeze(1)
        wd = wd.unsqueeze(1)
        return Ia * wa + Ib * wb + Ic * wc + Id * wd

    def differentiable_dewarping(self, src_fea, src_proj, ref_proj_inv, depth_samples):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]
        warped_src_fea = torch.zeros([batch, channels, height, width ],device=src_fea.device)
        with torch.no_grad():
            #inv = torch.inverse(ref_proj_inv)
            proj = torch.matmul(src_proj,ref_proj_inv)

            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                   torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            feature_xyz = xyz.long()
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            feature = src_fea #[B,C,H,W]

            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                                 height * width)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
            # avoid negative depth
            negative_depth_mask = proj_xyz[:, 2:3] <= 1e-3
            proj_xyz[:, 0:1][negative_depth_mask] = width
            proj_xyz[:, 1:2][negative_depth_mask] = height
            proj_xyz[:, 2:3][negative_depth_mask] = 1

            proj_xy = (proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :])
            proj_xy = proj_xy.long()
            negative_depth_mask = proj_xy[:, 0:1] <= 1e-3
            proj_xy[:, 0:1][negative_depth_mask] = int(0)
            negative_depth_mask = proj_xy[:, 1:2] <= 1e-3
            proj_xy[:, 1:2][negative_depth_mask] = int(0)
            negative_depth_mask = proj_xy[:, 0:1] > width - 1
            proj_xy[:, 0:1][negative_depth_mask] = width - 1
            negative_depth_mask = proj_xy[:, 1:2] > height - 1
            proj_xy[:, 1:2][negative_depth_mask] = height - 1

            nag = proj_xy<0

            if torch.any(nag):
                torch.save(proj_xy,"nagetive.pt")
                raise ValueError("has nagetive")
            for i in range(batch):
                warped_src_fea[i,:,proj_xy[i,1,0],proj_xy[i,0,0]] = feature[i,:,feature_xyz[1],feature_xyz[0]]



        warped_src_fea = warped_src_fea.view(batch, channels, height, width)

        return warped_src_fea

    def reproject(self, intr_mat, src_fea, intr_mat_inv, src_proj, ref_proj, depth_samples):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]

        with torch.no_grad():

            rot_src = src_proj[:, :3, :4]
            rot_ref = ref_proj[:, :3, :4]
            src_proj_ = torch.matmul(intr_mat_inv, rot_src)
            ref_proj_ = torch.matmul(intr_mat_inv, rot_ref)
            proj_ = torch.matmul(src_proj_[:, :3, :3], ref_proj_[:, :3, :3].transpose(1, 2))
            trans = torch.matmul(intr_mat, src_proj_[:, :3, 3:4] - torch.matmul(proj_, ref_proj_[:, :3, 3:4]))
            rot = torch.matmul(torch.matmul(intr_mat, proj_), intr_mat_inv)

            # proj = torch.matmul(src_proj,
            #                     torch.inverse(ref_proj))
            #
            # rot = proj[:, :3, :3]  # [B,3,3]
            # trans = proj[:, :3, 3:4]  # [B,3,1]

            y = torch.arange(0, height, dtype=torch.float32, device=src_fea.device).unsqueeze(1).repeat(1, width)
            x = torch.arange(0, width, dtype=torch.float32, device=src_fea.device).unsqueeze(0).repeat(height, 1)

            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(int(height * width)), x.view(int(height * width))
            xyz = torch.stack((x, y, torch.ones(x.size(), device="cuda")))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            # ###----------jiangpf-----------
            depth_samples = depth_samples.view(batch, num_depth, int(height * width))
            # xyz_depth = xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
            #                                                                                      height * width)
            # xyz_depth = xyz_depth.view(batch,3,int(num_depth * height * width))
            #
            # rot_depth_xyz = torch.matmul(rot, xyz_depth).view(batch, 3, num_depth, int(height * width))  # [B, 3, H*W]
            # proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                                 height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            mask = proj_xyz[:, 2:] > 1e-3

            proj_xyz *= mask
            mask = ~mask
            mask_xyz = torch.ones(mask.size(), device="cuda") * mask
            # mask_x = mask_xyz*0 #width
            # mask_y = mask_xyz*0 #height
            mask_z = mask_xyz * 1
            # mask_xyz = torch.cat((mask_x,mask_y,mask_z),dim=1)
            proj_xyz[:, 2:3] += mask_z
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            mask_x_max = proj_xy[:, 0:1] < width
            mask_x_min = proj_xy[:, 0:1] >= 0
            proj_xy[:, 0:1] *= mask_x_max
            proj_xy[:, 0:1] *= mask_x_min
            mask_y_max = proj_xy[:, 1:2] < height
            mask_y_min = proj_xy[:, 1:2] >= 0
            proj_xy[:, 1:2] *= mask_y_max
            proj_xy[:, 1:2] *= mask_y_min
            torch.cuda.synchronize(0)
            time2 = time.time()
            warped_src_fea = self.bilinear_interpolate_torch(src_fea, proj_xy[:, 1, :, :], proj_xy[:, 0, :, :])#.squeeze(2)
            warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        torch.cuda.synchronize(0)

        return warped_src_fea

    def differentiable_warping(self,src_fea, src_proj, ref_proj_inv, depth_samples):
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]

        with torch.no_grad():
            inv = torch.inverse(ref_proj_inv)
            proj = torch.matmul(src_proj,ref_proj_inv)


            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                   torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                                 height * width)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
            # avoid negative depth
            negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
            proj_xyz[:, 0:1][negative_depth_mask] = width
            proj_xyz[:, 1:2][negative_depth_mask] = height
            proj_xyz[:, 2:3][negative_depth_mask] = 1
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
            grid = proj_xy

        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)


        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        return warped_src_fea

    def get_depth(self,score,depth_sample):
        ret = []
        batch,channles,height,width = score.shape
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.int64, device=depth_sample.device),
                               torch.arange(0, width, dtype=torch.int64, device=depth_sample.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        idx = torch.argmax(score,dim=1)
        idx = idx.view(batch,height * width)
        xyz[:,2,:] = idx
        for i in range(batch):
           ret.append(depth_sample[i,xyz[i,2,:],xyz[i,1,:],xyz[i,0,:]].view(height,width))
        ret = torch.stack(ret)
        return ret
        
    
    def forward(self,eval_offset, ref_feature:torch.Tensor, src_features, ref_proj_inv, src_projs, depth_sample,
                        depth_min, depth_max, iter, offset=None, weight=None, view_weights=None):

        num_src_features = len(src_features)
        num_src_projs = len(src_projs)
        batch, feature_channel, height, width = ref_feature.size()
        device = ref_feature.get_device()

        src_offset = []
        
        num_depth = depth_sample.size()[1]
        assert num_src_features == num_src_projs, "Patchmatch Evaluation: Different number of images and projection matrices"
        #if view_weights != None:
            #assert num_src_features == view_weights.size()[1], "Patchmatch Evaluation: Different number of images and view weights"
        
        pixel_wise_weight_sum = 0
        
        ref_feature = ref_feature.view(batch, self.G, feature_channel//self.G, height, width)

        similarity_sum = 0
        torch.cuda.synchronize(0)
        time2_o = time.time()
        if self.stage == 3 and view_weights==None:
            view_weights = []
            torch.cuda.synchronize(0)
            time1 = time.time()

            for src_feature, src_proj in zip(src_features, src_projs):
                # batch, channels, height, width = src_feature.shape
                # num_depth = depth_sample.shape[1]
                # cdhw = torch.tensor([channels, num_depth, height, width], dtype=torch.float,
                #                     device=torch.device("cuda"))
                # warped_feature = reproject_(self.intr_mat.contiguous(), src_feature.contiguous(),
                #                             self.intr_mat_inv.contiguous(), src_proj.contiguous(),
                #                             ref_proj.contiguous(), depth_sample.contiguous(), cdhw.contiguous())
                # torch.save(self.intr_mat,"./checkpoints/intr_mat.pt")
                # torch.save(src_feature, "./checkpoints/src_feature.pt")
                # torch.save(self.intr_mat_inv, "./checkpoints/intr_mat_inv.pt")
                # torch.save(src_proj, "./checkpoints/src_proj.pt")
                # torch.save(ref_proj, "./checkpoints/ref_proj.pt")
                # torch.save(depth_sample, "./checkpoints/depth_sample.pt")
                # torch.save(cdhw, "./checkpoints/cdhw.pt")
                #
                #
                # torch.save(warped_feature, "./checkpoints/warped_feature.pt")
                # warped_feature = self.reproject(self.intr_mat.contiguous(), src_feature.contiguous(),
                #                             self.intr_mat_inv.contiguous(), src_proj.contiguous(),
                #                             ref_proj.contiguous(), depth_sample.contiguous())

                warped_feature = self.differentiable_warping(src_feature, src_proj, ref_proj_inv, depth_sample)
                #err_warp = torch.sum(torch.abs(warped_feature - warped_feature_))


                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                # group-wise correlation
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # pixel-wise view weight
                view_weight = self.pixel_wise_net(similarity)
                view_weights.append(view_weight)

                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) #[B,1,1,H,W]
                else:
                    similarity_sum += similarity*view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                #similarity_sum += similarity
                    
                #del warped_feature, src_feature, src_proj, similarity, view_weight
            torch.cuda.synchronize(0)
            time2 = time.time()

            #del src_features, src_projs
            view_weights = torch.cat(view_weights,dim=1) #[B,4,H,W], 4 is the number of source views
            # aggregated matching cost across all the source views
            similarity = similarity_sum.div_(pixel_wise_weight_sum)
            #del ref_feature, pixel_wise_weight_sum, similarity_sum
            # adaptive spatial cost aggregation
            score = self.similarity_net(similarity, offset, weight)
            #del similarity, offset, weight
            
            # apply softmax to get probability
            score = F.softmax(score)
            # softmax = nn.LogSoftmax(dim=1)
            # score = softmax(score)
            # score = torch.exp(score)
            
            # depth regression: expectation
            #depth_sample = torch.sum(depth_sample * score, dim = 1)
            depth_sample = self.get_depth(score, depth_sample)

            for src_feature, src_proj in zip(src_features, src_projs):
                offset_feature = self.differentiable_dewarping(eval_offset, src_proj, ref_proj_inv, depth_sample.unsqueeze(1))
                src_offset.append(offset_feature.detach())

            return depth_sample, score,view_weights.detach(),src_offset
        else:
            i=0
            torch.cuda.synchronize(0)
            time1 = time.time()
            for src_feature, src_proj in zip(src_features, src_projs):
                # batch, channels, height, width = src_feature.shape
                # num_depth = depth_sample.shape[1]
                # cdhw = torch.tensor([channels, num_depth, height, width], dtype=torch.float,
                #                     device=torch.device("cuda"))
                # warped_feature = reproject_(self.intr_mat.contiguous(), src_feature.contiguous(),
                #                             self.intr_mat_inv.contiguous(), src_proj.contiguous(),
                #                             ref_proj.contiguous(), depth_sample.contiguous(), cdhw.contiguous())

                warped_feature = self.differentiable_warping(src_feature, src_proj, ref_proj_inv, depth_sample)
                #err_warp = torch.sum(torch.abs(warped_feature - warped_feature_))


                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # reuse the pixel-wise view weight from first iteration of Patchmatch on stage 3
                #view_weight = self.pixel_wise_net(similarity)
                view_weight = view_weights[:,i].unsqueeze(1) #[B,1,H,W]
                i=i+1
                
                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) #[B,1,1,H,W]
                else:
                    similarity_sum += similarity*view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                #similarity_sum += similarity
                
                #del warped_feature, src_feature, src_proj, similarity, view_weight
            torch.cuda.synchronize(0)
            time2 = time.time()

            #del src_features, src_projs
                
            # [B, G, Ndepth, H, W]
            similarity = similarity_sum.div_(pixel_wise_weight_sum)
            
            #del ref_feature, pixel_wise_weight_sum, similarity_sum
            
            score = self.similarity_net1(similarity, offset, weight)
            score = F.softmax(score)

            # softmax = nn.LogSoftmax(dim=1)
            # score = softmax(score)
            # score = torch.exp(score)


            if self.stage == 1 and iter == self.iterations: 
                # depth regression: inverse depth regression
                depth_sample = self.get_depth(score, depth_sample)

                # depth_index = torch.arange(0, num_depth, 1, device=device).view(1, num_depth, 1, 1)
                # depth_index = torch.sum(depth_index * score, dim = 1)
                #
                # inverse_min_depth = 1.0 / depth_sample[:,-1,:,:] #sort emplication
                # inverse_max_depth = 1.0 / depth_sample[:,0,:,:] #sort emplication
                # depth_sample = inverse_max_depth + depth_index / (num_depth - 1) * \
                #                             (inverse_min_depth - inverse_max_depth)
                # depth_sample = 1.0 / depth_sample

                return depth_sample, score, src_offset
            
            # depth regression: expectation
            else:
                #depth_sample = torch.sum(depth_sample * score, dim = 1)
                depth_sample = self.get_depth(score, depth_sample)
                for src_feature, src_proj in zip(src_features, src_projs):
                    offset_feature = self.differentiable_dewarping(eval_offset, src_proj, ref_proj_inv,
                                                                   depth_sample.unsqueeze(1))
                    src_offset.append(offset_feature.detach())

                return depth_sample, score,src_offset


class PatchMatch(nn.Module):
    def __init__(self, random_initialization = False, propagation_out_range = 2, 
                patchmatch_iteration = 2, patchmatch_num_sample = 16, patchmatch_interval_scale = 0.025,
                num_feature = 64, G = 8, propagate_neighbors = 16, stage=3, evaluate_neighbors=9):
        super(PatchMatch, self).__init__()
        self.random_initialization = random_initialization
        self.depth_initialization = DepthInitialization(patchmatch_num_sample)
        self.depth_weight = Depth_weight(propagation_out_range)
        self.propagation_out_range = propagation_out_range
        self.propagation = Propagation(propagate_neighbors,propagation_out_range=propagation_out_range)
        self.patchmatch_iteration = patchmatch_iteration

        self.patchmatch_interval_scale = patchmatch_interval_scale
        self.propa_num_feature = num_feature
        # group wise correlation
        self.G = G

        #learn offset weight para
        # self.weight_prop = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.weight_eval = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.stage = stage
        
        self.dilation = propagation_out_range
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.evaluation = Evaluation(self.G, self.stage, self.evaluate_neighbors, self.patchmatch_iteration, propagation_out_range = propagation_out_range)
        self.evaluation1 = Evaluation(self.G, self.stage, self.evaluate_neighbors, self.patchmatch_iteration,
                                     propagation_out_range=propagation_out_range)
        # adaptive propagation
        if self.propagate_neighbors > 0:
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            if not (self.stage == 1 and self.patchmatch_iteration == 1):
                self.propa_conv = nn.Conv2d(
                                self.propa_num_feature,
                                self.propagate_neighbors * 2,
                                kernel_size=5,
                                stride=1,
                                padding=self.dilation*2,
                                dilation=self.dilation,
                                bias=True)

                nn.init.xavier_uniform_(self.propa_conv.weight)
                nn.init.constant_(self.propa_conv.bias, 0.)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(self.propa_num_feature, self.evaluate_neighbors * 2, kernel_size=5, stride=1,
                                    padding=self.dilation*2, dilation=self.dilation, bias=True)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.eval_conv.weight)
        nn.init.constant_(self.eval_conv.bias, 0.)
        self.feature_weight_net = FeatureWeightNet(num_feature, self.evaluate_neighbors, self.G, stage=self.stage,
                                                   propagation_out_range = propagation_out_range)


    # adaptive spatial cost aggregation
    # weight based on depth difference of sampling points and center pixel
    def depth_weight1(self,depth_sample, depth_min, depth_max,offset, patchmatch_interval_scale, evaluate_neighbors):
        # grid: position of sampling points in adaptive spatial cost aggregation
        neighbors = evaluate_neighbors
        batch, num_depth, height, width = depth_sample.size()
        _,cordin,hight,weight = offset.size()
        # normalization
        x = 1.0 / depth_sample
        #del depth_sample
        inverse_depth_min = 1.0 / depth_min
        inverse_depth_max = 1.0 / depth_max
        x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min.view(batch, 1, 1, 1) \
                                                            - inverse_depth_max.view(batch, 1, 1, 1))


        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d(int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range) * 2)
            # padding = nn.ConstantPad2d(int(pad_num - 1), 0)
            ref_feature_indx = padding(x[i])
            ref_feature_indx1 = padding1(x[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[:, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s + neighbors, :, :] * ref_feature_indx[:, indy:height + indy,
                                                                indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1) / 2

            ref_feature_list.append(ref_feature_sum)
        x1_ = torch.stack(ref_feature_list)

        x1_ = torch.abs(x1_ - x) / patchmatch_interval_scale
        x1_ = torch.clamp(x1_, min=0, max=4)
        x1_ = (-x1_ + 2) * 2
        output = nn.Sigmoid()
        x1_ = output(x1_)

        return x1_.detach()

    # adaptive spatial cost aggregation
    # weight based on depth difference of sampling points and center pixel
    def depth_weight2(self, depth_sample, depth_min, depth_max, offset, patchmatch_interval_scale,
                          evaluate_neighbors):
        # grid: position of sampling points in adaptive spatial cost aggregation
        neighbors = evaluate_neighbors
        batch, num_depth, height, width = depth_sample.size()
        # normalization
        x = 1.0 / depth_sample       ###-----jiangpf---------if depth-smaple is zero which needed modify
        #del depth_sample
        inverse_depth_min = 1.0 / depth_min
        inverse_depth_max = 1.0 / depth_max
        x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min.view(batch, 1, 1, 1)
                                                                - inverse_depth_max.view(batch, 1, 1, 1))


        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d(int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range) * 2 )
            # padding = nn.ConstantPad2d(int(pad_num - 1), 0)
            ref_feature_indx = padding(x[i])
            ref_feature_indx1 = padding1(x[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[:, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s + neighbors, :, :] * ref_feature_indx[:, indy:height + indy,
                                                                indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1) / 2

            ref_feature_list.append(ref_feature_sum)
        x1_ = torch.stack(ref_feature_list)

        ###-------jiangpf------
        # x1_ = self.conv_2d_feature1(x)
        # x1_ = x1_.view(batch, num_depth, height, width)
        x1_ = torch.abs(x1_ - x) / patchmatch_interval_scale
        x1_ = torch.clamp(x1_, min=0, max=4)
        x1_ = (-x1_ + 2) * 2
        output = nn.Sigmoid()
        x1_ = output(x1_)

        ###-------

        # del grid
        # x1 = x1.view(batch, num_depth, neighbors, height, width)

        # [B,Ndepth,N_neighbors,H,W]
        # x1 = torch.abs(x1 - x.unsqueeze(2)) / patchmatch_interval_scale
        # del x
        # x1 = torch.clamp(x1, min=0, max=4)
        #  # sigmoid output approximate to 1 when x=4
        # x1 = (-x1 + 2) * 2
        # output = nn.Sigmoid()
        # x1 = output(x1)

        return x1_.detach()

    def forward(self, ref_feature, src_features, ref_proj_inv, src_projs, depth_min, depth_max,
                depth = None,view_weights=None):
        depth_samples = []

        device = ref_feature.get_device()
        batch, _, height, width = ref_feature.size()

        # the learned additional 2D offsets for adaptive propagation
        if (self.propagate_neighbors > 0) and not (self.stage == 1 and self.patchmatch_iteration == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)

            propa_offset_ = self.propa_conv(ref_feature)
            propa_offset1 = F.softmax(propa_offset_[:, :self.propagate_neighbors])#version 1.2.1
            propa_offset2 = F.softmax(propa_offset_[:, self.propagate_neighbors:])
            #propa_offset = propa_offset1 + propa_offset2
            propa_offset = torch.cat([propa_offset1,propa_offset2],dim=1)
    
        # the learned additional 2D offsets for adaptive spatial cost aggregation (adaptive evaluation)

        eval_offset_ = self.eval_conv(ref_feature)
        #eval_offset_ = 1 / (1 + torch.exp(-1 * eval_offset_))
        eval_offset1 = F.softmax(eval_offset_[:, :self.evaluate_neighbors]) #version 1.2.3
        eval_offset2 = F.softmax(eval_offset_[:, self.evaluate_neighbors:])
        #eval_offset = eval_offset1 + eval_offset2
        eval_offset = torch.cat([eval_offset1,eval_offset2],dim=1)

        feature_weight,_ = self.feature_weight_net(ref_feature.detach(), eval_offset)  # eval_offset
        
        # first iteration of Patchmatch
        iter = 1
        hw = torch.tensor(( height, width), device=0)#depth_min[0], depth_max[0],
        if self.random_initialization:
            # first iteration on stage 3, random initialization, no adaptive propagation

            depth_sample = self.depth_initialization(True, depth_min,depth_max,hw,
                                    self.patchmatch_interval_scale)

            # weights for adaptive spatial cost aggregation in adaptive evaluation

            weight = self.depth_weight(depth_sample.detach(), depth_min, depth_max, eval_offset.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)


            weight = weight * feature_weight.unsqueeze(1)  #feature and depth position weight
            #weight = weight / torch.sum(weight, dim=2).unsqueeze(2)  #sum deformable conv
            
            # evaluation, outputs regressed depth map and pixel-wise view weights which will
            # be used for subsequent iterations

            depth_sample, score, view_weights,src_offset = self.evaluation(eval_offset_,ref_feature, src_features, ref_proj_inv, src_projs,
                                        depth_sample, depth_min, depth_max, iter, eval_offset, weight, view_weights)

            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
        else:
            # subsequent iterations, local perturbation based on previous result

            depth_sample = self.depth_initialization(False, depth_min,depth_max,hw, self.patchmatch_interval_scale, depth)

            #del depth

            # adaptive propagation
            if (self.propagate_neighbors > 0) and not (self.stage == 1 and iter == self.patchmatch_iteration):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                #if not (self.stage == 1 and iter == self.patchmatch_iteration):

                depth_sample = self.propagation(depth_sample, propa_offset)


            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = self.depth_weight(depth_sample.detach(), depth_min, depth_max, eval_offset.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            
            # evaluation, outputs regressed depth map
            depth_sample, score ,src_offset= self.evaluation(eval_offset_,ref_feature, src_features, ref_proj_inv, src_projs,
                                        depth_sample, depth_min, depth_max, iter, eval_offset, weight, view_weights)
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)

        for iter in range(2, self.patchmatch_iteration+1):
            torch.cuda.synchronize(0)
            time3_1 = time.time()
            # local perturbation based on previous result
            depth_sample = self.depth_initialization(False, depth_min,depth_max,hw, self.patchmatch_interval_scale, depth_sample)
            torch.cuda.synchronize(0)
            time3_2 = time.time()
            # adaptive propagation
            if (self.propagate_neighbors > 0) and not (self.stage == 1 and iter == self.patchmatch_iteration):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)

                depth_sample = self.propagation(depth_sample, propa_offset)



            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = self.depth_weight(depth_sample.detach(), depth_min, depth_max, eval_offset.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            # weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            torch.cuda.synchronize(0)
            time3_3 = time.time()
            # evaluation, outputs regressed depth map
            depth_sample, score,src_offset = self.evaluation1(eval_offset_,ref_feature, src_features,
                                                ref_proj_inv, src_projs, depth_sample, depth_min, depth_max, iter, eval_offset, weight, view_weights)
            torch.cuda.synchronize(0)
            time3_4 = time.time()

            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
            src_offset.append(eval_offset)

        return depth_samples, score,view_weights,src_offset
        

# first, do convolution on aggregated cost among all the source views
# second, perform adaptive spatial cost aggregation to get final cost
class SimilarityNet(nn.Module):
    def __init__(self, G, neighbors = 9, stage = 3, propagation_out_range =2 ):
        super(SimilarityNet, self).__init__()
        self.neighbors = neighbors
        self.propagation_out_range = propagation_out_range - 1
        
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.stage = stage
        
    def forward(self, x1, offset, weight):
        # x1: [B, G, Ndepth, H, W], aggregated cost among all the source views with pixel-wise view weight
        # grid: position of sampling points in adaptive spatial cost aggregation
        # weight: weight of sampling points in adaptive spatial cost aggregation, combination of 
        # feature weight and depth weight
        
        batch,G,num_depth,height,width = x1.size()
        batch,neighbors,_,_ = offset.size()
        neighbors = int(neighbors/2)
        x1 = self.similarity(self.conv1(self.conv0(x1))).view(batch,num_depth,height,width)

        ###-------jiangpf-------

        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d(int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range) * 2)
            # padding = nn.ConstantPad2d(int(pad_num - 1), 0)
            ref_feature_indx = padding(x1[i])
            ref_feature_indx1 = padding1(x1[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1) * 2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[:, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s + neighbors, :, :] * ref_feature_indx[:, indy:height + indy,
                                                                indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1) / 2

            ref_feature_list.append(ref_feature_sum)
        x1_ = torch.stack(ref_feature_list)
        
        # [B,Ndepth,9,H,W]
        # x1 = x1.view(batch, num_depth, self.neighbors, height, width)
    
        return x1_*weight # torch.sum(x1*weight, dim=2)

# adaptive spatial cost aggregation
# weight based on similarity of features of sampling points and center pixel
class FeatureWeightNet(nn.Module):
    def __init__(self, num_feature, neighbors=9, G=8, stage=3,device = 0,propagation_out_range=2):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = G
        self.stage = stage
        self.device = device
        self.propagation_out_range = propagation_out_range - 1

        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        ###----------jiangpf-------------------
        self.conv0_pf = ConvBnReLU(G,16,1,1,0)
        self.conv1_pf = ConvBnReLU(16,8,1,1,0)
        self.similarity_pf = nn.Conv2d(8,1,1,1,0)

        ###--------------------
        
        self.output = nn.Sigmoid()

    def forward(self, ref_feature, offset): #grid,
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation
        batch,feature_channel,height,width = ref_feature.size()
        batch,neighbors,_,_ = offset.size()
        neighbors = int(neighbors/2)
        ###--------jiangpf------------
        ref_feature = ref_feature.view(batch, feature_channel, height, width)
        # with torch.no_grad():
        #     weight_ = self.conv_2d_feature.weight
        #     mask = torch.zeros((9,9),device=)
        #     weight_*offset
        #     F.conv2d(ref_feature,)

        ref_feature_list = []
        for i in range(batch):
            pad_num = np.sqrt(neighbors)
            padding = nn.ReflectionPad2d (int(pad_num / 2) + self.propagation_out_range)
            padding1 = nn.ReflectionPad2d((int(pad_num / 2) + self.propagation_out_range)*2)
            #padding = nn.ConstantPad2d(int(pad_num - 1), 0)
            ref_feature_indx = padding(ref_feature[i])
            ref_feature_indx1 = padding1(ref_feature[i])
            ref_feature_sum = 0
            for s in range(neighbors):
                indy1 = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)*2
                indx1 = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)*2
                ref_feature_1 = offset[i, s, :, :] * ref_feature_indx1[:, indy1:height + indy1, indx1:width + indx1]
                indy = int(s / np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                indx = int(s % np.sqrt(neighbors)) * (self.propagation_out_range + 1)
                ref_feature_ = offset[i, s+neighbors, :, :] * ref_feature_indx[:, indy:height + indy, indx:width + indx]
                ref_feature_sum += (ref_feature_ + ref_feature_1)/2

            ref_feature_list.append(ref_feature_sum)
        ref_feature = torch.stack(ref_feature_list)

        # x_ = self.conv_2d_feature(ref_feature)
        ref_feature_ = ref_feature.view(batch, self.G, feature_channel // self.G, height, width)
        # x_ = x_.view(batch, self.G, feature_channel // self.G, height, width)
        # x_ = (x_ * ref_feature_).mean(2)
        x_ = ref_feature_.mean(2)
        x_ = self.similarity_pf(self.conv1_pf(self.conv0_pf(x_)))
        x_ = x_.view(batch, height, width)
        ###-----------

        return self.output(x_),ref_feature


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, G):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv2.weight)
        

    def forward(self, x1):
        # x1: [B, G, Ndepth, H, W]
        batch, G, ndepth, h, w = x1.size()
        # [B, Ndepth, H, W]
        x1 =self.conv2(self.conv1(self.conv0(x1))).view(batch,ndepth,h,w)
        
        output = self.output(x1)
        #del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]
        
        return output.unsqueeze(1)
        
            # if self.stage == 0:
            #     intr_mat = torch.tensor([[2892.33008,0,823.20398],[0,2883.16992,619.07001],[0,0,1]],device= 0)
            #     intr_mat_inv = torch.tensor([[3.4574e-4, 0, -0.28461620],[0, 3.4684e-4, -0.21471854],[0, 0, 1]],
            #                                 device=0)
            # elif self.stage == 1:
            #     intr_mat = torch.tensor([[1446.16504,0,411.60199],[0,1441.58496,309.53500],[0,0,1]],device=0)
            #     intr_mat_inv = torch.tensor([[6.9148e-4, 0, -0.28461620],[0, 6.9368e-4, -0.21471854],[0, 0, 1]],
            #                                 device=0)
            # elif self.stage == 2:
            #     intr_mat = torch.tensor([[723.08252,0,205.80099],[0,720.79248,154.76750],[0,0,1]],device=0)
            #     intr_mat_inv = torch.tensor([[1.38297e-3, 0, -0.28461620],[0, 1.38736e-3, -0.21471854],[0, 0, 1]],
            #                                 device=0)
            # elif self.stage == 3:
            #     intr_mat = torch.tensor([[361.54126,0,102.90050],[0,360.39624,77.38375],[0,0,1]],device=0)
            #     intr_mat_inv = torch.tensor([[2.76594e-3,0,-0.28461620],[0,2.77472e-3,-0.21471854],[0,0,1]],device=0)
            # rot_src = src_proj[:, :3, :4]
            # rot_ref = ref_proj[:, :3, :4]
            # src_proj_ = torch.matmul(intr_mat_inv,rot_src)
            # ref_proj_ = torch.matmul(intr_mat_inv,rot_ref)
            # proj_ = torch.matmul(src_proj_[:,:3,:3],ref_proj_[:,:3,:3].transpose(1, 2))
            # trans =torch.matmul(intr_mat,src_proj_[:,:3,3:4] - torch.matmul(proj_,ref_proj_[:,:3,3:4]))
            # rot = torch.matmul(torch.matmul(intr_mat,proj_),intr_mat_inv)


class Reproject(Function):
    @staticmethod
    def forward(ctx,intr_mat,src_fea,intr_mat_inv, src_proj, ref_proj, depth_samples,cdhw):  # ctx 必须要
        # src_fea: [B, C, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # depth_samples: [B, Ndepth, H, W]
        # out: [B, C, Ndepth, H, W]
        batch, channels, height, width = src_fea.shape
        num_depth = depth_samples.shape[1]


        with torch.no_grad():

            rot_src = src_proj[:, :3, :4]
            rot_ref = ref_proj[:, :3, :4]
            src_proj_ = torch.matmul(intr_mat_inv, rot_src)
            ref_proj_ = torch.matmul(intr_mat_inv, rot_ref)
            proj_ = torch.matmul(src_proj_[:, :3, :3], ref_proj_[:, :3, :3].transpose(1, 2))
            trans = torch.matmul(intr_mat, src_proj_[:, :3, 3:4] - torch.matmul(proj_, ref_proj_[:, :3, 3:4]))
            rot = torch.matmul(torch.matmul(intr_mat, proj_), intr_mat_inv)


            y = torch.arange(0, height, dtype=torch.float32, device=src_fea.device).unsqueeze(1).repeat(1, width)
            x = torch.arange(0, width, dtype=torch.float32, device=src_fea.device).unsqueeze(0).repeat(height, 1)

            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(int(height * width)), x.view(int(height * width))
            xyz = torch.stack((x, y, torch.ones(x.size(), device="cuda")))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

            depth_samples = depth_samples.view(batch, num_depth, int(height * width))

            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                                 height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            mask = proj_xyz[:, 2:] > 1e-3

            proj_xyz *= mask
            mask = ~mask
            mask_xyz = torch.ones(mask.size(), device="cuda") * mask

            mask_z = mask_xyz * 1

            proj_xyz[:, 2:3] += mask_z
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            mask_x_max = proj_xy[:, 0:1] < width
            mask_x_min = proj_xy[:, 0:1] >= 0
            proj_xy[:, 0:1] *= mask_x_max
            proj_xy[:, 0:1] *= mask_x_min
            mask_y_max = proj_xy[:, 1:2] < height
            mask_y_min = proj_xy[:, 1:2] >= 0
            proj_xy[:, 1:2] *= mask_y_max
            proj_xy[:, 1:2] *= mask_y_min

            warped_src_fea = bilinear_interpolate_torch(src_fea, proj_xy[:, 1, :, :], proj_xy[:, 0, :, :])
            warped_src_fea = warped_src_fea.squeeze(2)
            # warped_src_fea = F.grid_sample(src_fea, proj_xy.view(batch, num_depth * height, width, 2), mode='bilinear',
            #                                padding_mode='zeros', align_corners=True)
            warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        return warped_src_fea

    @staticmethod
    @parse_args('v','v','v','v','v','v','v')
    def symbolic(g, intr_mat, src_feature, intr_mat_inv, src_proj, ref_proj, depth_sample, cdhw):

        return g.op("custom::Reproject", intr_mat, src_feature, intr_mat_inv, src_proj, ref_proj, depth_sample, cdhw
                    )#.setType(src_feature.type().with_dtype(torch.float32).with_sizes(src_feature_shape))



def bilinear_interpolate_torch(im, y, x):
    '''
       im : B,C,H,W
       y : 1,numPoints -- pixel location y float
       x : 1,numPOints -- pixel location y float
    '''
    batch, _, _, _ = im.size()
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.cuda.LongTensor)
    y1 = y0 + 1

    wa = (x1.type(torch.cuda.FloatTensor) - x) * (y1.type(torch.cuda.FloatTensor) - y)
    wb = (x1.type(torch.cuda.FloatTensor) - x) * (y - y0.type(torch.cuda.FloatTensor))
    wc = (x - x0.type(torch.cuda.FloatTensor)) * (y1.type(torch.cuda.FloatTensor) - y)
    wd = (x - x0.type(torch.cuda.FloatTensor)) * (y - y0.type(torch.cuda.FloatTensor))
    # Instead of clamp
    n1 = x1 / im.shape[3]
    n2 = y1 / im.shape[2]
    x1 = x1 - torch.floor(n1).int()
    y1 = y1 - torch.floor(n2).int()
    Ia = []
    Ib = []
    Ic = []
    Id = []
    for i in range(batch):
        Ia.append(im[i:i + 1, :, y0[i], x0[i]])
        Ib.append(im[i:i + 1, :, y1[i], x0[i]])
        Ic.append(im[i:i + 1, :, y0[i], x1[i]])
        Id.append(im[i:i + 1, :, y1[i], x1[i]])
    Ia = torch.cat(Ia, dim=0)
    Ib = torch.cat(Ib, dim=0)
    Ic = torch.cat(Ic, dim=0)
    Id = torch.cat(Id, dim=0)
    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

reproject_ = Reproject.apply