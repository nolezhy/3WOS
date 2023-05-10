import lightgbm
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _safe_indexing
from process.fisvdd import fisvdd
import base_sampler
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import random
import time

cat_data = np.load(r'D:\pycharm\pythonProject\3WOS\data\OpticalDigits.npz')
all_data = cat_data['data']
all_label = cat_data['label']
kf = KFold(n_splits=5, shuffle=True)
right = 0
test_len = 0
TP = 0
FN = 0
FP = 0
TN = 0
start_time = time.time()
# min_data,maj_data= base_sampler.seperate_minor_and_major_data(cat_data)

for i in range(all_data.shape[1]):
    # 对第i列(第i个特征)进行最值归一化
    if np.max(all_data[:, i]) == 0 and np.min(all_data[:, i]) == 0:
        continue
    else:
        all_data[:, i] = (all_data[:, i] - np.min(all_data[:, i])) / (np.max(all_data[:, i]) - np.min(all_data[:, i]))


index=np.where(all_label==1)
reserve_index = np.where(all_label == -1)
wait_data = all_data[index]  # 获取了少数类数据
minnum=wait_data.shape[0]
#wait_data1 = train_data_all[index]
reserve_data = all_data[reserve_index]  # 获取了多数类的数据
maxnum=reserve_data.shape[0]

weight=round(maxnum/minnum)
#print(weight)
#print('minnum,maxnum',minnum,maxnum)
k=minnum*3

print(k)
#sampler = fisvdd(data=wait_data, sigma=0.4)  # ---调整抽取的比例sigma就是样本数据的大小
#wait_data1 = train_data_all[index]
#reserve_data1 = train_data_all[reserve_index]
#sv_index = sampler.find_sv()
#print(sv_index.shape[0], reserve_data.shape[0])#抽取少数类个数，多数类的个数
#sample_data = wait_data[sv_index]  # 获取抽取的少数类样本点的数据
nn_min_data = NearestNeighbors(n_neighbors=6).fit(wait_data).kneighbors(wait_data,
                                                                            return_distance=False)[:, 1:]

#diff = reserve_data.shape[0] - wait_data.shape[0]
diff=k

samples_indices = np.random.randint(low=0, high=np.shape(wait_data)[0], size=diff)#从0-少数类数量的数据中抽取diff个数据(比如从0-10中取10个数)
steps = np.random.uniform(size=diff)#返回0-1之间的浮点数diff个
cols = np.mod(samples_indices, nn_min_data.shape[1])
reshaped_feature = np.zeros((diff, wait_data.shape[1]))
for i, (col, step) in enumerate(zip(cols, steps)):#合成多少类差值的数据
    row = samples_indices[i]
    reshaped_feature[i] = wait_data[row] - step * (
            wait_data[row] - wait_data[nn_min_data[row, col]])
    new_min_feature_data = np.vstack((reshaped_feature, wait_data))#把合成的数据跟所有少数类放在一起形成一个平衡的数据

min_label = 1
new_labels_data = np.array([min_label] * np.shape(new_min_feature_data)[0])
new_minor_data_arr2 = np.column_stack((new_min_feature_data, new_labels_data))
max_label = -1
reserve_labels_data = np.array([max_label] * np.shape(reserve_data)[0])
new_max_data_arr2 = np.column_stack((reserve_data, reserve_labels_data))
balanced_data_arr2 = base_sampler.concat_and_shuffle_data(new_minor_data_arr2, new_max_data_arr2)
#print(balanced_data_arr2)
balanced_data=balanced_data_arr2[:,:-1]
balanced_label=balanced_data_arr2[:,-1]
# ave_F = 0
# ave_G = 0
# ave_A = 0
# ave_R = 0
# ave_P = 0
# ave_C=0
# ave_T=0
# ave_FN=0
# ave_FP=0
for train_index, test_index in kf.split(balanced_data):  # 5次验证的(训练数据：训练标签；测试数据：测试标签)=(4:1)
    train_data = balanced_data[train_index]
    train_label = balanced_label[train_index]
    test_data = balanced_data[test_index]
    test_label = balanced_label[test_index]
    clf = SVC(C=1.0, class_weight=None, coef0=0.0, decision_function_shape='ovr', gamma='auto', kernel='rbf',
             max_iter=-1, random_state=None, tol=0.0001)
    # clf.fit(train_data,train_label)
   #  clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=0)
   #  clf.fit(train_data, train_label)
    #clf = MultinomialNB()
   #  clf = LogisticRegression()
    clf.fit(train_data, train_label)


    pred_y = clf.predict(test_data)
    result = pred_y == test_label

    for i in range(len(test_data)):
        if pred_y[i]==1 and test_label[i] == 1:
            TP += 1
        elif pred_y[i]== -1 and test_label[i] == -1:
            TN += 1
        elif pred_y[i]== -1 and test_label[i] == 1:
            FN += 1
        else:
            FP += 1
precision = TP / (TP + FP)  # 正例中有多少被预测为正例
recall = TP / (TP + FN)  # TPR#Senisitivity#正例被判成了正例有多少
Specificity = TN / (TN + FP)  # 负类被正确分类TNR
    # TPR=TP/TP+FN#Senisitivity#正例被判成了正例
FPR = FP / (TN + FP)  # 多少负例被判成了正例
cost = 4
totalcost = cost * FN + 1 * FP
F1 = (2 * precision * recall) / (precision + recall)
correct = (TP + TN) / (TP + FP + TN + FN)
G_mean = (recall * Specificity) ** 0.5
end_time = time.time()
d_time = end_time - start_time
    # ave_P = precision + ave_P
    # ave_R = recall + ave_R
    # ave_F = F1 + ave_F
    # ave_G = G_mean + ave_G
    # ave_A = correct + ave_A
    # ave_C = ave_C + totalcost
    # ave_T = ave_T + d_time
    # ave_FN=FN+ave_FN
    # ave_FP=FP+ave_FP
print('FP,FN,TN,TP', FP, FN, TN, TP)
print("*Specificity", Specificity)
print("*totalcost", totalcost)
print("*G_mean:", G_mean)
print("F1 Score:", F1)
print('recall,precision', recall, precision)
print("correct:", correct)
print("run time:", d_time, "s")
print("********************************")

# print('ave_A',ave_A/5)
# print('ave_G', ave_G / 5)
# print('ave_F', ave_F / 5)
# print('ave_P', ave_P / 5)
# print('ave_R', ave_R / 5)
# print('ave_C', ave_C / 5)
# print('ave_T', ave_T / 5)
# print('ave_FN',ave_FN/5)
# print('ave_FP',ave_FP/5)
