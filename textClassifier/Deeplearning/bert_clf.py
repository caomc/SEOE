import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertTokenizer, BertModel

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
from sklearn.model_selection import train_test_split
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertModel.from_pretrained("bert-base-chinese")
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
for i in range(len(train_X)) :
    print(i)
    data=train_X[i].split(" ")
    temp=[0 for j in range(768)]
    res=torch.tensor(temp).float()
    res=res.unsqueeze(0)
    for j in data:
        input_ids = torch.tensor(tokenizer.encode(j)).unsqueeze(0)
        op=model(input_ids)[1]
        res+=op
    train_X[i]=torch.nn.functional.normalize(res)
for i in range(len(X_test)) :
    print(i)
    data=val_X[i].split(" ")
    temp=[0 for j in range(768)]
    res=torch.tensor(temp).float()
    res=res.unsqueeze(0)
    for j in data:
        input_ids = torch.tensor(tokenizer.encode(j)).unsqueeze(0)
        op=model(input_ids)[1]
        res+=op
    val_X[i]=torch.nn.functional.normalize(res)
for i in range(len(X_test)):
    print(i)
    data = X_test[i].split(" ")
    temp = [0 for j in range(768)]
    res = torch.tensor(temp).float()
    res = res.unsqueeze(0)
    for j in data:
        input_ids = torch.tensor(tokenizer.encode(j)).unsqueeze(0)
        op = model(input_ids)[1]
        res += op
    X_test[i] = torch.nn.functional.normalize(res)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,Y_train)
pre=knn.predict(X_test)
print(pre)