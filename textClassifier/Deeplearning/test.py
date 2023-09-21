import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

def read_data():
    print('Reading data......')
    label = np.load('../data/input/label.npy')
    data = []

    with open('../data/input/ga_document.seq') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(line)
    print(label.shape)
    print(len(data))
    return data, label


data, label = read_data()
X_train, X_test, Y_train, Y_true = train_test_split(data,label,test_size=0.2, random_state=0, stratify=label)

np.savetxt('../data/output/label.txt',label)
#with open('../data/output/label.txt', 'a') as file:
#   np.savetxt(label,file)