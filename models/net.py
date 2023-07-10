import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .patchmatch import *
from .add_position import feature_add_position

class FeatureNet(nn.Module):
    def __init__(self,num_features = [8, 16, 32, 64]):
        super(FeatureNet, self).__init__()
        self.feature_channels = num_features[0]

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        #self.conv0 = ConvBn(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        #self.conv1 = ConvBn(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

        torch.nn.init.orthogonal(self.output1.weight)
        torch.nn.init.orthogonal(self.inner1.weight)
        torch.nn.init.orthogonal(self.inner2.weight)
        torch.nn.init.orthogonal(self.output2.weight)
        torch.nn.init.orthogonal(self.output3.weight)

    def forward(self, x):
        features = []

        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)

        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature3 = self.output1(conv10)

        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7, conv10
        output_feature2 = self.output2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(
                conv4)
        del conv4
        output_feature1 = self.output3(intra_feat)

        del intra_feat

        return output_feature3, output_feature2, output_feature1
        

class Refinement(nn.Module):
    def __init__(self):
        
        super(Refinement, self).__init__()
        
        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        
        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8)
        self.res = nn.Conv2d(8, 1, 3, padding=1, bias=False)
        torch.nn.init.orthogonal(self.deconv.weight)
        torch.nn.init.orthogonal(self.res.weight)
        
        
    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0-depth_min.view(batch_size,1,1,1))/(depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1))
        
        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = F.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1)) + depth_min.view(batch_size,1,1,1)

        return depth

class Patch(nn.Module):
    def __init__(self, patchmatch_interval_scale = [0.005, 0.0125, 0.025], propagation_range = [6,4,2],
                patchmatch_iteration = [1,2,2], patchmatch_num_sample = [8,8,16], propagate_neighbors = [0,8,16],
                evaluate_neighbors = [9,9,9],n_views=5):
        super(Patch, self).__init__()
        num_features = [8, 16, 32, 64]
        self.stages = 4
        self.G = [4,8,8]
        self.propagate_neighbors = propagate_neighbors
        self.n_views = n_views
        for l in range(self.stages - 1):

            if l == 2:
                patchmatch = PatchMatch(True, propagation_range[l], patchmatch_iteration[l],
                                        patchmatch_num_sample[l], patchmatch_interval_scale[l],
                                        num_features[l + 1], self.G[l], self.propagate_neighbors[l], l + 1,
                                        evaluate_neighbors[l])
            else:
                patchmatch = PatchMatch(False, propagation_range[l], patchmatch_iteration[l],
                                        patchmatch_num_sample[l], patchmatch_interval_scale[l],
                                        num_features[l + 1], self.G[l], self.propagate_neighbors[l], l + 1,
                                        evaluate_neighbors[l])
            setattr(self, f'patchmatch_{l + 1}', patchmatch)

        #-----------------------------#
        self.Linear2 = nn.Conv2d(evaluate_neighbors[0]*2, num_features[2], kernel_size=1)
        self.Linear1 = nn.Conv2d(evaluate_neighbors[0]*2, num_features[1], kernel_size=1)

    def forward(self,src_features,ref_feature,depth_min,proj_matrices,depth_max):
        self.proj_matrices_0 = torch.unbind(proj_matrices['stage_0'].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices['stage_1'].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices['stage_2'].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices['stage_3'].float(), 1)
        depth = None
        view_weights = None
        src_offset = None
        depth_patchmatch = {}
        for l in reversed(range(1, self.stages)):
            src_features_l = [src_fea[l-1] for src_fea in src_features]

            projs_l = getattr(self, f'proj_matrices_{l}')
            ref_proj, src_projs = projs_l[0], projs_l[1:]
            ref_proj_inv = torch.inverse(ref_proj)

            if l == 3:
                depth, score, view_weights,src_offset = self.patchmatch_3(ref_feature[l-1], src_features_l,
                                                 ref_proj_inv, src_projs,
                                                 depth_min, depth_max,view_weights=view_weights)

            elif l == 2:
                #-----------------#
                out = self.Linear2(src_offset[self.n_views - 1])
                ref_feature[l-1] += out
                for idx, val in enumerate(src_features_l):
                    out = self.Linear2(src_offset[idx])
                    src_features_l[idx] += out

                depth, score, view_weights,src_offset = self.patchmatch_2(ref_feature[l-1], src_features_l,
                                                 ref_proj_inv, src_projs,
                                                 depth_min, depth_max, depth=depth,view_weights=view_weights)

            elif l == 1:
                #-------------------#
                out = self.Linear1(src_offset[self.n_views - 1])
                ref_feature[l-1] += out
                for idx, val in enumerate(src_features_l):
                    out = self.Linear1(src_offset[idx])
                    src_features_l[idx] += out

                depth, score, view_weights,src_offset = self.patchmatch_1(ref_feature[l-1], src_features_l,
                                                 ref_proj_inv, src_projs,
                                                 depth_min, depth_max, depth=depth,view_weights=view_weights)

            depth_patchmatch[f'stage_{l}'] = depth
            del src_features_l, ref_proj, src_projs, projs_l

            depth = depth[-1].detach()
            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth,
                                      scale_factor=2, mode='nearest')
                view_weights = F.interpolate(view_weights,
                                    scale_factor=2, mode='nearest')
                for idx,val in enumerate(src_offset):
                    src_offset[idx] = F.interpolate(src_offset[idx],
                                        scale_factor=2, mode='nearest')
        return depth,score,depth_patchmatch

class NormalsLoss(nn.Module):
    def forward(self, normals_gt_b3hw, normals_pred_b3hw):

        normals_mask_b1hw = torch.logical_and(
            normals_gt_b3hw.isfinite().all(dim=1, keepdim=True),
            normals_pred_b3hw.isfinite().all(dim=1, keepdim=True))

        normals_pred_b3hw = normals_pred_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        normals_gt_b3hw = normals_gt_b3hw.masked_fill(~normals_mask_b1hw, 1.0)

        with torch.cuda.amp.autocast(enabled=False):
            normals_dot_b1hw = 0.5 * (
                                        1.0 - torch.einsum(
                                                "bchw, bchw -> bhw",
                                                normals_pred_b3hw,
                                                normals_gt_b3hw,
                                            )
                                        ).unsqueeze(1)
        normals_loss = normals_dot_b1hw.masked_select(normals_mask_b1hw).mean()

        return normals_loss

class AdaptMVSNet(nn.Module):
    def __init__(self, patchmatch_interval_scale = [0.005, 0.0125, 0.025], propagation_range = [6,4,2],
                patchmatch_iteration = [1,2,2], patchmatch_num_sample = [8,8,16], propagate_neighbors = [0,8,16],
                evaluate_neighbors = [9,9,9],n_views=5):
        super(AdaptMVSNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.normals_loss = NormalsLoss()

        self.patchmatch_num_sample = patchmatch_num_sample
        
        num_features = [8, 16, 32, 64]
        
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4,8,8]

        self.patch = Patch(patchmatch_interval_scale, propagation_range,
                patchmatch_iteration, patchmatch_num_sample, propagate_neighbors,
                evaluate_neighbors,n_views=n_views)

        self.upsample_net = Refinement()

    def depth_regression(self,p, depth_values):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
        depth = torch.sum(p * depth_values, 1)
        depth = depth.unsqueeze(1)
        return depth

    def forward(self, imgs,proj_matrices, depth_min, depth_max) : #imgs,proj_matrices, depth_min, depth_max
        
        imgs_0 = torch.unbind(imgs['stage_0'], 1)
        imgs_1 = torch.unbind(imgs['stage_1'], 1)
        imgs_2 = torch.unbind(imgs['stage_2'], 1)
        imgs_3 = torch.unbind(imgs['stage_3'], 1)
        
        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        
        # step 1. Multi-scale feature extraction
        features = []

        for img in imgs_0:
            output_feature3,output_feature2,output_feature1 = self.feature(img)

            output_feature = []
            output_feature.append(output_feature1)
            output_feature.append(output_feature2)
            output_feature.append(output_feature3)

            features.append(output_feature)

        ref_feature, src_features = features[0], features[1:]


        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None

        depth_patchmatch = {}
        refined_depth = {}

        depth,score,depth_patchmatch = self.patch(src_features,ref_feature ,depth_min,proj_matrices,depth_max) #depth_patchmatch

        torch.cuda.synchronize(device=0)
        time3 = time.time()
        # step 3. Refinement
        
        if self.training:
            depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
            refined_depth['stage_0'] = depth

            del depth, ref_feature, src_features
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch,
                    }
            
        else:
            depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
            # return depth,score
            refined_depth['stage_0'] = depth

            del depth, ref_feature, src_features
            num_depth = self.patchmatch_num_sample[0]
            batch, channels, height, width = score.shape
            pad_in = F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2))
            score_sum4 = 4 * F.avg_pool3d(pad_in, (4, 1, 1), stride=1, padding=0)
            score_sum4 = score_sum4.view(batch, channels, height, width)
            # [B, 1, H, W]
            depth_index = self.depth_regression(score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)).long()
            depth_index = torch.clamp(depth_index, 0, num_depth-1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence,
                                        scale_factor=2, mode='nearest')
            batch, channels, height, width = photometric_confidence.shape
            photometric_confidence = photometric_confidence.view(batch,height,width)
            # return refined_depth,photometric_confidence
            return {"refined_depth": refined_depth, #[1,1,1200,1600]
                        "depth_patchmatch": depth_patchmatch, #don't need
                        "photometric_confidence": photometric_confidence,#[1,1200,1600]
                    }

    def adaptmvsnet_loss(self,depth_patchmatch, refined_depth, depth_gt,normals_gt,normals_pred, mask):

        stage = 4

        loss = 0
        for l in range(1, stage):
            depth_gt_l = depth_gt[f'stage_{l}']
            mask_l = mask[f'stage_{l}'] > 0
            depth2 = depth_gt_l[mask_l]

            depth_patchmatch_l = depth_patchmatch[f'stage_{l}']

            for i in range(len(depth_patchmatch_l)):
                depth1 = depth_patchmatch_l[i][mask_l]
                loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

        for l in range(stage):
            #if l == 0 or l == 1:
                normals_loss = self.normals_loss(normals_gt[f'stage_{l}'], normals_pred[f'stage_{l}'])
                loss = loss + normals_loss
        l = 0
        depth_refined_l = refined_depth[f'stage_{l}']
        depth_gt_l = depth_gt[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0

        depth1 = depth_refined_l[mask_l]
        depth2 = depth_gt_l[mask_l]
        loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

        return loss

def adaptmvsnet_loss(depth_patchmatch, refined_depth, depth_gt, mask):

    stage = 4

    loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    l = 0
    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    return loss
