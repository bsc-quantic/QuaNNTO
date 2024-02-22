import numpy as np

def trigonometric_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            transf_feats[i,j] = (j+1)*np.sin((j+1)*features[i])
    return transf_feats

def polynomial_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            transf_feats[i,j] = (j+1)*features[i]**(j+1)
    return transf_feats

def get_range(data):
    return (np.min(data), np.max(data))

def rescale_data(data, data_range, scale_data_range):
    data_range_dist = data_range[1] - data_range[0]
    norm_data_range_dist = scale_data_range[1] - scale_data_range[0]
    return scale_data_range[0] + norm_data_range_dist * (data - data_range[0]) / data_range_dist
