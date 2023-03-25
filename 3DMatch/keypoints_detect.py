import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
# function: get key point(salient point)
# input：
#     data: input data(np.array)
#     gamma21: parameter1
#     gamma32: parameter1
#     KDTree_radius: the radius of KDTree 
#     NMS_radius: the radius of NMS
#     max_num: max point number
# output：
#     keypoints_after_NMS: key points after NMS
def iss(data, gamma21, gamma32, KDTree_radius, NMS_radius, max_num=100):
    leaf_size = 32
    tree = KDTree(data, leaf_size)
    radius_neighbor = tree.query_ball_point(data, KDTree_radius)
    keypoints = [] 
    min_feature_value = []  
    for index in range(len(radius_neighbor)):
        neighbor_idx = radius_neighbor[index]
        neighbor_idx.remove(index) 
        if len(neighbor_idx)==0:
            continue

        weight = np.linalg.norm(data[neighbor_idx] - data[index], axis=1)
        weight[weight == 0] = 0.001 
        weight = 1 / weight

        cov = np.zeros((3, 3))
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]
        for i in range(len(neighbor_idx)):
            cov += weight[i]*tmp[i].dot(tmp[i].transpose())
        cov /= np.sum(weight)

        '''
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]  # N,3,1
        cov = np.sum(weight[:, np.newaxis, np.newaxis] *
                     (tmp @ tmp.transpose(0, 2, 1)), axis=0) / np.sum(weight)
        '''

        s = np.linalg.svd(cov, compute_uv=False)    
      
        if (s[1]/(s[0]+0.000001) < gamma21) and (s[2]/(s[1]+0.000001) < gamma32):
            keypoints.append(data[index])
            min_feature_value.append(s[2])

    # NMS step
    keypoints_after_NMS = []
    leaf_size = 10 
    nms_tree = KDTree(keypoints, leaf_size)
    index_all = [i for i in range(len(keypoints))]
    for iter in range(max_num):
        max_index = min_feature_value.index(max(min_feature_value))
        tmp_point = keypoints[max_index]
        del_indexs = nms_tree.query_ball_point(tmp_point, NMS_radius)
        for del_index in del_indexs:
            if del_index in index_all:
                del min_feature_value[index_all.index(del_index)]   
                del keypoints[index_all.index(del_index)]           
                del index_all[index_all.index(del_index)]          
        keypoints_after_NMS.append(tmp_point)
        if len(keypoints) == 0:
            break

    return np.array(keypoints_after_NMS)

if __name__=="__main__":
    pts=np.rand([1000,3])
    keypoint = iss(pts, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.3, max_num=5000)

