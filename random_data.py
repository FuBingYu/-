import math
import numpy as np
import random
import sys

#data processing
tags = np.zeros((1000,1),dtype=int)
spam_count = 0
sum = 1000#训练的邮件总数
characters = 5
with open("C:/Users/傅冰玉/Desktop/自然语言处理/朴素贝叶斯/a.txt","w") as f:
    # data为[1000,6]的二维数组，一共1000条数据，5个属性，最后一列为标签
    train_data = np.random.randint(0,2,size=[sum,characters+1])
    for i in range(sum):
        #如果是垃圾邮件，标签为0
        if train_data[i][-1] == 0:
            tags[i] == 0
            spam_count +=1
        np.set_printoptions(threshold=sys.maxsize)#将数据完整输出
        f.write(str(train_data)+"\n")
f.close()
#print(spam_count)#垃圾邮件的数量
normal_count = sum - spam_count
#exit()


# different characters conditional P
def Cal_condintional_P(tag,list,num,sum):#标签，待检测的序列，标签的总数，数据的总数
    p = 1.0
    for c in range(4):
        count = 0.0
        # 拉普拉斯修正可以有效避免因训练样本不充分而导致概率估值为0的问题。
        # 但本次实验中，随机生成的由于属性的维度较低，在所有数据中肯定出现过已经给定的属性组合，所以不需要使用拉普拉斯修正
        #count = 1.0
        #num = num+2
        for i in range(sum):
            if (train_data[i][c] ==list[c]) and (train_data[i][4]==tag):
                count += 1.0
        p = p*(count/num)
        #print(p)
    p = p*num/sum
    #p = p * (num+1) / (sum+2)
    return p

'''P1_x=Cal_condintional_P(1,[0,1,0,1],spam_count,sum)
P0_x=Cal_condintional_P(0,[0,1,0,1],normal_count,sum)
print(P0_x)
print(P1_x)'''

def Cal_accuracy():
    test_data = np.random.randint(0,2,size=[200,characters+1])
    test_label = np.zeros((200,1),dtype=int)
    n = 0.0
    for i in range(200):
        con0 = Cal_condintional_P(1,test_data[i],spam_count,sum)
        con1 = Cal_condintional_P(0,test_data[i],normal_count,sum)
        if con0 > con1:
            test_label[i]=0

        if test_label[i] == test_data[i][-1]:
            n += 1.0

    acc = n / 200.0
    return acc


acc = Cal_accuracy()
print("准确度为：",acc)
