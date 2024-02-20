import numpy as np

def trigonometric_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            # TODO: CHECK THE VALIDNESS OF THIS INPUT (SQUEEZING FACTORS RESTRICTED TO > 0)
            #transf_feats[i,j] = 2*(j+1)*np.arcsin(features[i])
            transf_feats[i,j] = (j+1)*np.sin((j+1)*features[i])
    return transf_feats

def get_range(data):
    return (np.min(data), np.max(data))

def normalize_data(data, data_range, norm_data_range):
    data_range_dist = data_range[1] - data_range[0]
    norm_data_range_dist = norm_data_range[1] - norm_data_range[0]
    return norm_data_range[0] + norm_data_range_dist * (data - data_range[0]) / data_range_dist

def denormalize_data(norm_data, data_range, norm_data_range):
    data_range_dist = data_range[1] - data_range[0]
    norm_data_range_dist = norm_data_range[1] - norm_data_range[0]
    return data_range[0] + data_range_dist * (norm_data - norm_data_range[0]) / norm_data_range_dist