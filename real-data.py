import math

# 1 读取训练数据
NUM = 5000    #     邮件总数，包括训练集和测试集
spam_num = 0        #     统计垃圾邮件个数
train_labels = []      #     记录训练邮件的0/1，因为一会要集合去重，去除1/0
train_sen_words = []              #   用来保存每一封训练邮件中的单词
words = []                #   本次训练集生成的词库

train_file = open(r"C:/Users/傅冰玉/Desktop/naive_bayes-master/train.txt")
read = train_file.readlines()
for line in read:
    te = line.rstrip('\n').split(' ')  #完成分词
    for i in te:
        i = i.lower()
    words += te        #  存储词库,需要前面的0/1标签，予以保留
    if te[0] == '0':
        spam_num += 1
        train_labels.append(0)
    else:
        train_labels.append(1)
    del(te[0])
    train_sen_words.append(list(set(te)))   #去除训练集中重复单词，去除了0/1标签,保存每一封邮件的单词
train_file.close()

# 生成新的词库（去掉words中重复的单词,停用词）
new_words =set(words)
stopword_file = open(r"C:/Users/傅冰玉/Desktop/naive_bayes-master/stop.txt", "r", encoding='utf-8')
word_list = []
for word in stopword_file.readlines():
    word_list.extend(word.strip().split(' '))
stopwords = set(word_list)
second_new_words = []
for i in range(len(new_words)):
    for j in new_words:
        if j in stopwords:
            continue
    else:
        second_new_words.append(j)

# 2 处理训练数据
la_s = list(new_words)
la_s.remove('1')
la_s.remove('0')
val = []             #  邮件中出现所有的单词对应的计数列表
for i in range(len(la_s)):
    val.append(0)
spam_count = dict(zip(la_s,val))  #垃圾邮件中的计数
normal_count = dict(zip(la_s,val))   #正常邮件中的计数

count = 0
for aa in train_sen_words:#遍历训练集，找出每一个单词在垃圾邮件和非垃圾邮件是否出现
    for i in aa:
        if i in new_words:
           if train_labels[count] == 0:
               spam_count[i] += 1
           else:
               normal_count[i] += 1
    count +=1
''' 有可能单词只在垃圾邮件或正常邮件中出现，那么可能出现乘0
 此时需要拉普拉斯修正，可以有效避免因训练样本不充分而导致概率估值为0的问题。'''
for i in la_s:
    if spam_count[i] == 0 or normal_count[i] == 0:
        spam_count[i] += 1
        normal_count[i] += 1

# 3 读取测试数据
test_labels = []        #     记录测试邮件的0/1，因为一会要集合去重，去除1/0
test_file = open(r"C:/Users/傅冰玉/Desktop/naive_bayes-master/test.txt")
test_sen_words = []         #    用来保存每一封测试邮件中的单词
read = test_file.readlines()
for line in read:
    te = line.rstrip('\n').split()
    for i in te:#转小写
        i = i.lower()
    if te[0] == '0':
        test_labels.append(0)
    else:
        test_labels.append(1)
    del(te[0])
    test_sen_words.append(list(set(te)))#去除测试集中重复单词

# 4 对测试数据进行预测
normal_num = NUM - spam_num
Ps = spam_num / NUM
Pn = 1 - Ps
count = 0
cal_test_labels = []

for i in test_sen_words:
    #垃圾邮件
    ts = math.log(Ps, 10)
    #正常邮件
    tn = math.log(Pn, 10)
    for j in i:
        # 如果j这个单词在字典里，加上它出现的频率（log相加，等于相乘）
        if j in new_words:
            ts += math.log(spam_count[j], 10)
            tn += math.log(normal_count[j], 10)
            # 如果训练集中垃圾邮件没有出现j这个单词，减去它出现的频率，
            # 注意：拉普拉斯修正
            if spam_count[j] == 1:
                ts -= math.log(spam_num + 1, 10)
            else:
                ts -= math.log(spam_num, 10)
            if normal_count[j] == 1:
                tn -= math.log(normal_num + 1, 10)
            else:
                tn -= math.log(normal_num, 10)
    # 根据计算的条件概率判断标签
    if ts > tn:
        cal_test_labels.append(0)
    if ts < tn:
        cal_test_labels.append(1)

# 5 计算分类器的准确率
count =0.0
sum = len(cal_test_labels)
for i in range(sum):
    if cal_test_labels[i]== int(test_labels[i]):
        count += 1.0
acc = count/sum
print("准确率是：",acc)

# 6 预测未标记的邮件
# 6.1 处理未标记的邮件
f_b = []#记录测试邮件的0/1，因为一会要集合去重，去除1/0
un_file = open(r"C:/Users/傅冰玉/Desktop/naive_bayes-master/unlabelled.txt")
un_sen_words = []#来存储测试数据
read = un_file.readlines()
for line in read:
    te = line.rstrip('\n').split()
    for i in te:#转小写
        i = i.lower()
    un_sen_words.append(list(set(te)))#去除测试集中重复单词
un_file.close()
# 6.2 对未标记的邮件进行预测
ct = 0
cal_un_labels = []
for i in un_sen_words:
    #垃圾邮件
    ts = math.log(Ps, 10)
    #正常邮件
    tn = math.log(Pn, 10)
    for j in i:
        # 如果j这个单词在字典里，加上它出现的频率（log相加，等于相乘）
        if j in new_words:
            ts += math.log(spam_count[j], 10)
            tn += math.log(normal_count[j], 10)
            # 如果训练集中垃圾邮件没有出现j这个单词，减去它出现的频率，
            # 注意：拉普拉斯修正
            if spam_count[j] == 1:
                ts -= math.log(spam_num + 1, 10)
            else:
                ts -= math.log(spam_num, 10)
            if normal_count[j] == 1:
                tn -= math.log(normal_num + 1, 10)
            else:
                tn -= math.log(normal_num, 10)
    # 根据计算的条件概率判断标签
    if ts > tn:
        cal_un_labels.append(0)
    if ts < tn:
        cal_un_labels.append(1)
# 6.3 将预测结果写入一个文件
with open(r"C:/Users/傅冰玉/Desktop/naive_bayes-master/ans.txt",'w') as f:
    for x in cal_un_labels:
        f.write(str(x))
        f.write('\n')
print("预测完成，已写入ans.txt文件。")
