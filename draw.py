import lkh
import matplotlib.pyplot as plt
import numpy as np

classes=['Positive','Negative']

# 标签的个数

tp=956
fp=95
fn=23
tn=78
# 在标签中的矩阵
confusion_matrix = np.array([
    (tp,fp),
    (fn,tn),
    ],dtype=np.int)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title('SEOE')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(2)] for i in range(2)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center')   #显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()
print("confusion matrix:")

print(tp, fp)
print(fn, tn)
print("acc1:",tp/(tp+fp))
print("acc0:",tn/(tn+fn))
pre=(tp)/(tp+fp)
rec=(tp)/(tp+fn)
print("pre:",pre)
print("recall:",rec)
print("F1:",2*pre*rec/(pre+rec))
2*(tp)/(tp+fp)*(tp)/(tp+fn)/(tp*tn/(tp+fp)(tn+fn))

lkh.solve(solver='LKH', problem=)