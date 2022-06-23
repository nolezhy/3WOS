import numpy as np
from sklearn.neighbors import NearestNeighbors


def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data).kneighbors(old_feature_data,
                                                                                      return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)#1表示多数类的label
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)