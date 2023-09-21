import pandas as pd
import numpy as np
import logging
import gensim
import umap
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.metrics import f1_score

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, MaxPooling1D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.initializers import *
from keras.models import *
from keras.models import load_model
from keras import backend as K
#from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *
import tensorflow as tf
import os
import time
import gc
import re
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()


#设置随机种子保证可重复性
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()


# 划分训练集、测试集
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
print(len(X_train), len(X_test), len(Y_train), len(Y_true))

train_X, val_X, train_y, val_y = train_test_split(X_train,Y_train,test_size=0.2, random_state=0, stratify=Y_train)
print(len(train_X), len(val_X), len(train_y), len(val_y))
print(train_X[0])

# 超参数
embed_size = 768
#embed_size = 768 # how big is each word vector
max_features = 38 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 25 # max number of words in a question to use #99.99%

## Tokenize the sentences
#生成词典
tokenizer = Tokenizer(num_words=max_features, filters='')
tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

#可以将每个string的每个词转成数字
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
X_test = tokenizer.texts_to_sequences(X_test)
print(train_X[0])

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
print(train_X[0])

del data, label
gc.collect()

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

def embedding_bert():
    a = {'p02_1': "年龄是十多岁", 'p02_2': "年龄是二十多岁", 'p02_3': "年龄是三十多岁", 'p02_4': "年龄是四十多岁", 'p02_5': "年龄是五十多岁",
         'p02_6': "年龄是六十多岁", 'p02_7': "年龄是七十多岁",
         'p03_0': "是少数民族", 'p03_1': "是汉族", 'p04_1': "职业是工人", 'p04_2': "职业是农民", 'p04_3': "职业是干部", 'p04_4': "职业是医护人员",
         'p04_5': "职业是教师", 'p04_6': "职业是职员", 'p04_7': "职业是服务员", 'p04_8': "职业是经商", 'p04_9': "职业是其他职业",
         'p05_1': "文化程度是硕士及以上", 'p05_2': "文化程度是本科", 'p05_3': "文化程度是专科", 'p05_4': "文化程度是高中及以下", 'p06_0': "不吸烟",
         'p06_1': "偶尔吸烟", 'p06_21': "每天吸烟小于10支", 'p06_22': "每天吸烟10-30支", 'p06_23': "每天吸烟大于30支",
         'p07_0': "无被动吸烟", 'p07_1': "偶尔被动吸烟", 'p07_21': "被动吸烟每天小于3小时", 'p07_22': "被动吸烟每天大于3小时", 'p08_0': "无毒品接触",
         'p08_1': "有毒品接触", 'p09_0': "无饮酒", 'p09_1': "偶尔饮酒", 'p09_21': "", 'p09_22': "", 'p09_23': "",
         'p10_0': "无有毒有害物质暴露", 'p10_11': "有放射线暴露", 'p10_12': "有高温暴露", 'p10_13': "有噪音暴露", 'p10_14': "有铅暴露",
         'p10_15': "有汞暴露",
         'p10_16': "有农药暴露", 'p10_17': "有家中装修暴露", 'p10_18': "密切接触过猫狗等宠物暴露",
         'p11_1': "家庭月收入2000元以下", 'p11_2': "家庭月收入2000-5000元", 'p11_3': "家庭月收入5000-10000元", 'p11_4': "家庭月收入10000元以上",
         'p12_0': "无孕产史", 'p12_11': "正常分娩", 'p12_12': "有抛宫产", 'p12_13': "有人工流产，自然流产’", 'p12_14': "有宫外孕",
         'p12_15': "有早产",
         'p12_16': "有死胎死产", 'p12_17': "有出生缺陷", 'p12_18': "其他",
         'p13_0': "无疾病史", 'p13_11': "有心脏病史", 'p13_12': "有慢性肾炎史", 'p13_13': "有高血压史", 'p13_14': "有糖尿病史",
         'p13_15': "患过甲状腺疾病",
         'p14_0': "无患过传染性疾病", 'p14_11': "有肺结核病史", 'p14_12': "有病毒性肝炎病史", 'p14_13': "有风疹病史", 'p14_14': "有巨细胞病毒病史",
         'p14_15': "有性传播疾病病史",
         'p15_1': "有患过或曾经患过心理疾病", 'p15_2': "没有患过心理疾病", 'p15_3': "不能确定是否患过或曾经患过心理疾病", 'p16_0': "目前没有服用药物",
         'p16_1': "目前服用药物",
         'p17_0': "不是近亲结婚", 'p17_1': "是近亲结婚",
         'p18_0': "家族中无遗传病史", 'p18_11': "家族中有遗传性血友病史", 'p18_12': "家族中有先天心脏病糖尿病病史", 'p18_13': "家族中有先天智力低下病史",
         'p18_14': "家族中有盲病史", 'p18_15': "家族中有聋病史", 'p18_16': "家族中有哑病史", 'p18_17': "家族中有精神病史", 'p18_2': "不确定是否有家族病史",
         'p19_1': "要宝宝是自己愿望", 'p19_2': "要宝宝是爱人的愿望", 'p19_3': "要宝宝是夫妻共同的愿望", 'p19_4': "要宝宝是父母的愿望",
         'p19_5': "要宝宝是别的家庭和朋友给的压力",
         'p20_1': "认为怀孕前要做好身体的准备", 'p20_2': "认为怀孕前要做好经济的准备", 'p20_3': "认为怀孕前要做好心理的准备", 'p20_4': "认为怀孕前要做好生育知识的准备",
         'p21_1': "已做好做父母的心理准备", 'p21_2': "未做好做父母的心理准备", 'p21_3': "不确定是否做好做父母的心理准备",
         'p22_1': "我认为孩子会影响到我和爱人的关系", 'p22_2': "我认为孩子不会影响到我和爱人的关系", 'p22_3': "不确定孩子是否会影响到我和爱人的关系", 'p23_1': "很在意孩子的性别",
         'p23_2': "不在意孩子的性别", 'p23_3': "不确定是否在意孩子的性别",
         'p24_1': "对于怀孕最担心的是宝宝是否健康", 'p24_2': "对于怀孕最担心的是养育宝宝的经济压力", 'p24_3': "对于怀孕最担心的是怀孕带给自己的不适",
         'p24_4': "对于怀孕最担心的是分娩痛苦",
         'p24_5': "对于怀孕最担心的是生宝宝对自己工作和事业带来影响", 'p24_6': "对于怀孕最担心的是日后宝宝会占用自己的私人空间", 'p24_7': "对于怀孕最担心的是生宝宝体型变化",
         'p25_1': "认为夫妻双方或一方精神高度紧张会影响受孕", 'p25_2': "认为夫妻双方或一方精神高度紧张不会影响受孕", 'p25_3': "不确定夫妻双方或一方精神高度紧张是否会影响受孕",
         'p26_1': "过度紧张和焦虑恐惧会引起流产", 'p26_2': "过度紧张和焦虑恐惧不会引起流产", 'p26_3': "不确定过度紧张和焦虑恐惧是否会引起流产",
         'p27_1': "认为不良的精神刺激和恐惧焦虑抑郁情绪在妊娠早期会影响胎儿生长发育", 'p27_2': "认为不良的精神刺激和恐惧焦虑抑郁情绪在妊娠早期会不影响胎儿生长发育",
         'p27_3': "不确定不良的精神刺激和恐惧焦虑抑郁情绪在妊娠早期是否会影响胎儿生长发育",
         'p28_1': "精神紧张会导致宫外孕的发生", 'p28_2': "精神紧张不会导致宫外孕的发生", 'p28_3': "不确定精神紧张是否会导致宫外孕的发生", 'p29_1': "目前感觉有压力",
         'p29_2': "目前感觉没有压力", 'p29_3': "不确定是否有压力",
         'p30_1': "与爱人关系和睦", 'p30_2': "与爱人关系一般", 'p30_3': "与爱人关系紧张", 'p31_1': "与亲友关系和睦", 'p31_2': "与亲友关系一般",
         'p31_3': "与亲友关系紧张", 'p32_1': "与同事关系和睦", 'p32_2': "与同事关系一般", 'p32_3': "与同事关系紧张", 'p35_0': "了解孕前优生心理咨询与指导",
         'p35_1': "不了解孕前优生心理咨询与指导", 'p36_0': "计划怀孕前接受过优生心理咨询和指导", 'p36_1': "计划怀孕前没有接受过优生心理咨询和指导",
         'p37_1': "认为计划怀孕前有必要进行优生心理咨询和指导", 'p37_2': "认为计划怀孕前没必要进行优生心理咨询和指导", 'p37_3': "认为计划怀孕前无所谓进行优生心理咨询和指导",
         'p38_1': "有通过墙报专栏了解孕前优生心理咨询和指导", 'p38_2': "有通过文图宣传品了解孕前优生心理咨询和指导", 'p38_3': "有通过网络了解孕前优生心理咨询和指导",
         'p38_4': "有通过广播电视了解孕前优生心理咨询和指导", 'p38_5': "有通过报刊杂志了解孕前优生心理咨询和指导", 'p38_6': "有通过培训讲座了解孕前优生心理咨询和指导",
         'p38_7': "有通过听身边人介绍了解孕前优生心理咨询和指导", 'p39_1': "希望在计划生育服务站接受孕前优生心理咨询和指导",
         'p39_2': "希望在妇幼保健站接受孕前优生心理咨询和指导", 'p39_3': "希望在综合性医院接受孕前优生心理咨询和指导", 'p39_4': "希望在专业的心理咨询机构接受孕前优生心理咨询和指导",
         'p40_1': "喜欢单独咨询方式进行孕前优生心理咨询和指导", 'p40_2': "喜欢面对面咨询方式进行孕前优生心理咨询和指导",
         'p40_3': "喜欢电话方式进行孕前优生心理咨询和指导", 'p40_4': "喜欢团体培训方式进行孕前优生心理咨询和指导"}
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # model = BertModel.from_pretrained('./chinese_bert_wwm_ext_L-12_H-768_A-12')

    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    model = BertModel.from_pretrained("bert-base-chinese")
    embedding_matrix = []
    for i in a:
        #print(a[i])
        input_ids = torch.tensor(tokenizer.encode(a[i])).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        #print(outputs[1])
        #print(outputs[1].shape)
        embedding_matrix.append(outputs[1].squeeze(0).detach().numpy())
    embedding_matrix = np.array(embedding_matrix)
    #print(embedding_matrix.shape)
    return embedding_matrix

def embedding_w2c():
    print('running embedding_w2c......')
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec.load('../data/input/model_ga.model')
    print('Found %s word vectors.' % len(model.wv.key_to_index.items()))
    embeddings_index = {}
    for k, v in model.wv.key_to_index.items():
        word = k
        vector = model.wv[k]
        embeddings_index[word] = vector
    print(len(embeddings_index))
    # 存储所有 word_index 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embedding_matrix = np.zeros((len(word_index) + 1, model.vector_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    return embedding_matrix
def tsne():
    xtrain=[]
    data=[]
    sample = pd.read_excel('E:/SEOE/' + 'all.xlsx', sheet_name=0)
    y =np.array(pd.read_excel('E:/SEOE/' + 'all.xlsx', sheet_name=1)) .tolist()
    ytrain=[]
    for i in y:
        ytrain.append(i[0])
    y = np.array(sample).tolist()
    print(ytrain.count(1))

    for i in y:
        temp=np.zeros(768)
        for j in i:
            if(j==1):

                temp+=embedding_matrix[j]
        #print(temp)
        xtrain.append(temp)
    import matplotlib.pyplot as plt
    reducer = umap.UMAP(random_state=3407)
    X_trans = reducer.fit_transform(xtrain)
    print(X_trans.shape)

    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=ytrain, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset')
    plt.show()

embedding_matrix = embedding_bert()
#embedding_matrix =embedding_w2c()
#tsne()
max_features = len(embedding_matrix)
print(embedding_matrix.shape)


def train_pred(model, epochs=2):
    access_result = {}
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    val_threshold = []
    val_f1 = []
    total_time = 0
    for e in range(epochs):
        print('epoch:', e)
        start_time = time.time()
        history = model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        end_time = time.time()
        total_time = total_time +(end_time-start_time)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        search_result = threshold_search(val_y, pred_val_y)
        print(search_result)
        # 保存threshold, f1
        threshold = [e, search_result['threshold']]
        val_threshold.append(threshold)
        f1 = [e, search_result['f1']]
        val_f1.append(f1)
        # threshold = 0.5
        # 保存loss,accuracy
        t_loss = [e, history.history['loss'][0]]
        train_loss.append(t_loss)
        v_loss = [e, history.history['val_loss'][0]]
        val_loss.append(v_loss)
        t_acc = [e, history.history['accuracy'][0]]
        train_accuracy.append(t_acc)
        v_acc = [e, history.history['val_accuracy'][0]]
        val_accuracy.append(v_acc)
    pred_test_y = model.predict([X_test], batch_size=1024, verbose=0)
    pred_train_y = model.predict([train_X], batch_size=1024, verbose=0)

    # print(val_threshold)
    access_result['train_loss'] = train_loss
    access_result['train_accuracy'] = train_accuracy
    access_result['val_loss'] = val_loss
    access_result['val_accuracy'] = val_accuracy
    access_result['val_threshold'] = val_threshold
    access_result['val_f1'] = val_f1
    avg_time = total_time/epochs
    print('average time:', avg_time)

    return pred_val_y, pred_test_y, pred_train_y, access_result


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.001 for i in range(250,500)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


def plot_result(access_result):
    # 画loss图
    train_loss = np.array(access_result['train_loss'])
    x1 = train_loss[:, 0]
    y1 = train_loss[:, 1]
    print(x1)
    val_loss = np.array(access_result['val_loss'])
    x2 = val_loss[:, 0]
    y2 = val_loss[:, 1]
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.legend(['train', 'val'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.title('Train History')
    plt.show()
    # 画accuracy图
    train_accuracy = np.array(access_result['train_accuracy'])
    x3 = train_accuracy[:, 0]
    y3 = train_accuracy[:, 1]
    val_accuracy = np.array(access_result['val_accuracy'])
    x4 = val_accuracy[:, 0]
    y4 = val_accuracy[:, 1]
    plt.plot(x3, y3)
    plt.plot(x4, y4)
    plt.legend(['train', 'val'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'accuracy')
    plt.title('Train History')
    plt.show()
    # 画threshold_f1图
    val_threshold = np.array(access_result['val_threshold'])
    x5 = val_threshold[:, 0]
    y5 = val_threshold[:, 1]
    val_f1 = np.array(access_result['val_f1'])
    x6 = val_f1[:, 0]
    y6 = val_f1[:, 1]
    plt.plot(x5, y5)
    plt.plot(x6, y6)
    plt.legend(['threshold', 'f1'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'threshold_f1')
    plt.title('Train History')
    plt.show()


# cnn2D卷积   Model1
def model_cnn2D(embedding_matrix):
    print('enter cnn model')
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
print("cnn")
pred_val_y1, pred_test_y1, pred_train_y1, access_result = train_pred(model_cnn2D(embedding_matrix), epochs=32)
print(pred_test_y1)
plot_result(access_result)
threshold = 0.42
y_true = Y_true
y_predict = pred_test_y1 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true, y_predict, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true, y_predict, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true, y_predict)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true, y_predict)))

y_true1 = train_y
y_predict1 = pred_train_y1 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true1, y_predict1, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true1, y_predict1, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true1, y_predict1)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true1, y_predict1)))


# Lstm + Gru    Model2
def model_lstm_gru(embedding_matrix):
    print("build model 2")
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(GRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
print("no clr")
pred_val_y2, pred_test_y2, pred_train_y2, access_result = train_pred(model_lstm_gru(embedding_matrix), epochs=41)
plot_result(access_result)
threshold = 0.253
y_true = Y_true
y_predict = pred_test_y2 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true, y_predict, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true, y_predict, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true, y_predict)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true, y_predict)))
#
y_true1 = train_y
y_predict1 = pred_train_y2 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true1, y_predict1, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true1, y_predict1, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true1, y_predict1)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true1, y_predict1)))


# CLR学习率策略
class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300., mode='exp_range',
               gamma=0.99994)


def train_pred2(model, epochs=2):
    access_result = {}
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    val_threshold = []
    val_f1 = []
    total_time = 0
    for e in range(epochs):
        print('clr')
        print('epoch:',e)
        start_time = time.time()
        history = model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks=[clr])
        end_time = time.time()
        total_time = total_time + (end_time - start_time)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        search_result = threshold_search(val_y, pred_val_y)
        print(search_result)
        # 保存threshold, f1
        threshold = [e, search_result['threshold']]
        val_threshold.append(threshold)
        f1 = [e, search_result['f1']]
        val_f1.append(f1)
        # 保存loss,accuracy
        t_loss = [e, history.history['loss'][0]]
        train_loss.append(t_loss)
        v_loss = [e, history.history['val_loss'][0]]
        val_loss.append(v_loss)
        t_acc = [e, history.history['accuracy'][0]]
        train_accuracy.append(t_acc)
        v_acc = [e, history.history['val_accuracy'][0]]
        val_accuracy.append(v_acc)
    pred_test_y = model.predict([X_test], batch_size=1024, verbose=0)
    pred_train_y = model.predict([train_X], batch_size=1024, verbose=0)

    print(val_threshold)
    access_result['train_loss'] = train_loss
    access_result['train_accuracy'] = train_accuracy
    access_result['val_loss'] = val_loss
    access_result['val_accuracy'] = val_accuracy
    access_result['val_threshold'] = val_threshold
    access_result['val_f1'] = val_f1
    avg_time = total_time / epochs
    print('average time:', avg_time)

    return pred_val_y, pred_test_y, pred_train_y, access_result
print("lstm gru clr")
pred_val_y7, pred_test_y7, pred_train_y7, access_result= train_pred2(model_lstm_gru(embedding_matrix), epochs=41)
plot_result(access_result)
threshold = 0.253
y_true = Y_true
y_predict = pred_test_y7 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true, y_predict, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true, y_predict, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true, y_predict)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true, y_predict)))

y_true1 = train_y
y_predict1 = pred_train_y7 > threshold
print('1. The F-1 score of the model {}\n'.format(f1_score(y_true1, y_predict1, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true1, y_predict1, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true1, y_predict1)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true1, y_predict1)))