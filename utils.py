import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):
    # 加载用户-物品交互数据
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)
    # 获取用户和物品的最大ID
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    
    u2i_index = [[] for _ in range(n_users + 1)] # 用户到物品的索引列表
    i2u_index = [[] for _ in range(n_items + 1)] # 物品到用户的索引列表

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# 负采样
def random_neq(l, r, s):
    t = np.random.randint(l, r) # 在[l, r]范围内随机生成整数
    while t in s:
        t = np.random.randint(l, r)  # 如果t在s中则重新生成
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # 为每个用户生成训练样本
    def sample(uid):
        # 确保用户至少有2个交互物品
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32) # 输入序列
        pos = np.zeros([maxlen], dtype=np.int32) #正样本标签
        neg = np.zeros([maxlen], dtype=np.int32) # 负样本标签

        nxt = user_train[uid][-1] # 最后一个交互物品作为第一个预测目标
        idx = maxlen - 1 # 从序列末尾开始填充
        ts = set(user_train[uid]) # 转为集合

        # 逆序遍历（排除最后一个）
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i 
            pos[idx] = nxt  # 正样本（即下一个物品）
            neg[idx] = random_neq(1, itemnum + 1, ts)  # 用户未交互过的物品
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids) # 每遍历完一轮用户后打乱数据
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch)) # 将批次数据放入队列（使用zip进行转置：将多个(u,s,p,n)列表转换为(u列表, s列表, p列表, n列表)）


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True  # 设置守护进程
            self.processors[-1].start()  # 启动进程

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()  # 终止进程
            p.join()       # 等待进程结束


# train/val/test 数据集划分
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)  # 用户交互历史字典
    user_train = {}
    user_valid = {}
    user_test = {}
    # user/item的索引从1开始
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ') # 解析用户ID和物品ID
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])  # 用户交互数量
        if nfeedback < 4:  # 交互数量小于4，全部作为训练数据
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2] # 除最后2个交互外的交互作为训练数据
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # 倒数第二个作为验证数据
            user_test[user] = []
            user_test[user].append(User[user][-1])  # 倒数第一个作为测试数据
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset) # 深拷贝，避免修改原始数据

    NDCG = 0.0  # 归一化折损累计增益
    HT = 0.0    # 命中率
    valid_user = 0.0  # 有效用户计数

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue  # 跳过没有足够数据的用户

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u]) # 已交互物品集合
        rated.add(0)          # 填充
        # 构建候选物品列表：1个正样本+100个负样本
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # 模型预测（argsort默认升序，故使用符号）
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # 获取预测分数

        rank = predictions.argsort().argsort()[0].item()  # 排名

        valid_user += 1

        if rank < 10:  # 如果排名在前10名内，更新指标
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
