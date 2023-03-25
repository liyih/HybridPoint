import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from geotransformer.modules.ops.pairwise_distance import pairwise_distance
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.utils.keypoints_detect import iss
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
import numpy as np
from backbone import KPConvFPN
import pdb
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans
def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
    - A:    [bs, num_corr, 3], source point cloud
    - B:    [bs, num_corr, 3], target point cloud
    - weights: [bs, num_corr]   weight for each correspondence 
    - weight_threshold: float,  clips points with weight below threshold
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)
def SM(corr, src_keypts, tgt_keypts, inlier_threshold=0.1, top_ratio=0.1):
    diff = corr - corr.permute(1, 0, 2)
    M = torch.sum(diff[:, :, 0:3] ** 2, dim=-1) ** 0.5 - torch.sum(diff[:, :, 3:6] ** 2, dim=-1) ** 0.5
    M = M[None, :, :]
    # pdb.set_trace() 
    ## polynomial funcition
    sigma = inlier_threshold / 3
    M = torch.max(torch.zeros_like(M), 4.5 - M ** 2 / 2 / sigma ** 2)
    M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
    
    # calculate the principal eigenvector
    leading_eig = torch.ones_like(M[:, :, 0:1])
    for _ in range(10):

        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
    leading_eig = leading_eig.squeeze(-1)
    # pdb.set_trace()
    # select top-10% as the inliers
    top_10p = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(leading_eig.shape[1] * top_ratio)]
    pred_labels = torch.zeros_like(leading_eig)
    pred_labels[0, top_10p[0]] = 1 # assert bs = 1

    # compute the transformation
    # pred_trans = rigid_transform_3d(src_keypts, tgt_keypts, leading_eig * pred_labels)
    return pred_labels
class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.in_project = nn.Linear(1024, 256)
        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a
        )
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )
        self.corr_num=cfg.coarse_matching.num_correspondences
        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        
    def forward(self, data_dict):
        output_dict = {}
                      
        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()
        ref_length_iss=data_dict['ref_length_iss']
        src_length_iss=data_dict['src_length_iss']
    
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        ###################################
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
    
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        dual_level=True

        with torch.no_grad():
            if dual_level==False:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                    ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
                )
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                # salient point matching
                ref_node_corr_indices_iss, src_node_corr_indices_iss, node_corr_scores_iss = self.coarse_matching(
                    ref_feats_c_norm[:ref_length_iss], src_feats_c_norm[:src_length_iss], ref_node_masks[:ref_length_iss], src_node_masks[:src_length_iss]
                )
                # non-salient point matching
                ref_node_corr_indices_sup, src_node_corr_indices_sup, node_corr_scores_sup = self.coarse_matching(
                    ref_feats_c_norm[ref_length_iss:], src_feats_c_norm[src_length_iss:], ref_node_masks[ref_length_iss:], src_node_masks[src_length_iss:]
                )
                ref_node_corr_indices_sup=ref_node_corr_indices_sup+ref_length_iss
                src_node_corr_indices_sup=src_node_corr_indices_sup+src_length_iss

                iss_select=int((ref_length_iss+src_length_iss)/(len(ref_feats_c_norm)+len(src_feats_c_norm))*self.corr_num)
                sup_select=self.corr_num-iss_select

                ref_keypt_sup=ref_points_c[ref_node_corr_indices_sup[:sup_select]].unsqueeze(0)
                src_keypt_sup=src_points_c[src_node_corr_indices_sup[:sup_select]].unsqueeze(0)
                corr = torch.cat([src_keypt_sup,ref_keypt_sup],dim=-1)
                #use SM on non-salient correspondences
                pred_labels=SM(corr, src_keypt_sup, ref_keypt_sup)
                if pred_labels.sum() == 0:
                    ref_node_corr_indices=ref_node_corr_indices_iss[:iss_select]
                    src_node_corr_indices=src_node_corr_indices_iss[:iss_select]
                    node_corr_scores=node_corr_scores_iss[:iss_select]
                # pdb.set_trace()
                else:
                    ref_node_corr_indices=torch.cat([ref_node_corr_indices_iss[:iss_select],ref_node_corr_indices_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)
                    src_node_corr_indices=torch.cat([src_node_corr_indices_iss[:iss_select],src_node_corr_indices_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)
                    node_corr_scores=torch.cat([node_corr_scores_iss[:iss_select],node_corr_scores_sup[:sup_select][pred_labels.bool().squeeze()]],dim=-1)
                
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            
            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]
            # pdb.set_trace()
            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
