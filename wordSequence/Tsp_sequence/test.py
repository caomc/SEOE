import numpy as np
from wordSequence.Tsp_sequence import functions

# 读入样本、属性
sample = np.load('../data/sample_line.npy')
feature = np.load('../data/feature_line.npy')

# 读入feature.name文件
f_name = [];
with open('../data/feature.name', 'r') as f_name_file:
    for line in f_name_file:
        #去掉开头结尾多余的空格
        f_name.append(line.strip())

nums = sample[0]
feature_name, distance_matrix = functions.feature(nums, f_name, feature)
#context = np.load('../data/sample_line.npy',encoding="latin1")
print("distance_matrix shape:"+str(distance_matrix.shape))
print("sample shape:"+str(sample.shape))
print("feature shape:"+str(feature.shape))
print(distance_matrix[0])
print(distance_matrix[1])