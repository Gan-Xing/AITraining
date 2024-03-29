import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
# 定义列数据类型：用户ID，电影ID和评分
dtype = [("userId", np.string_), ("productId", np.string_), ("Rating", np.int32)]
# 加载训练集数据，指定列数据类型
train_dataset = pd.read_csv('train.csv', dtype=dict(dtype))
# 显示前5行数据
train_dataset.head(5)
class LFM(object):
    # 初始化函数
    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10, columns=["uid", "iid", "Rating"]):
        self.alpha = alpha  # 学习率
        self.reg_p = reg_p  # P矩阵正则化系数
        self.reg_q = reg_q  # Q矩阵正则化系数
        self.number_LatentFactors = number_LatentFactors  # 隐因子的数量
        self.number_epochs = number_epochs  # 迭代次数
        self.columns = columns  # 数据集列名

    # 训练模型的函数
    def fit(self, dataset):
        # 将数据集转换为DataFrame格式
        self.dataset = pd.DataFrame(dataset)

        # 按用户ID和物品ID分组，聚合所有评分
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        # 计算全局平均评分
        self.globalMean = self.dataset[self.columns[2]].mean()

        # 使用随机梯度下降(SGD)进行优化
        self.P, self.Q = self.sgd()

    # 初始化P和Q矩阵
    def _init_matrix(self):
        # 根据用户和物品数量随机初始化P和Q矩阵
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    # 随机梯度下降优化
    def sgd(self):
        # 初始化P和Q
        P, Q = self._init_matrix()

        # 迭代训练
        for i in range(self.number_epochs):
            print("iter%d" % i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                # 获取用户和物品的隐因子向量
                v_pu = P[uid]  # 用户向量
                v_qi = Q[iid]  # 物品向量
                # 计算误差
                err = np.float32(r_ui - np.dot(v_pu, v_qi))

                # 更新P和Q矩阵
                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)
                
                P[uid] = v_pu
                Q[iid] = v_qi

                error_list.append(err ** 2)
            # 打印当前迭代的RMSE
            print(np.sqrt(np.mean(error_list)))
        return P, Q

    # 预测评分
    def predict(self, uid, iid):
        # 若用户或物品不在训练集中，返回全局平均评分
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]  # 获取用户的隐因子向量
        q_i = self.Q[iid]  # 获取物品的隐因子向量

    # 计算并返回预测评分
    return np.dot(p_u, q_i)

    # 测试模型
    def test(self, testset):
        '''对测试集进行评分预测'''
        # 创建空的DataFrame用于存储预测结果
        df = pd.DataFrame({"Keys": [], "Rating": []})
        i = 0
        # 遍历测试集中的每一条记录
        for uid, iid, _ in testset.itertuples(index=False):
            # 调用predict方法进行评分预测
            pred_rating = self.predict(uid, iid)
            # 四舍五入到最近的整数
            pred_rating = round(pred_rating)
            # 创建新行并追加到DataFrame
            new_row = pd.Series({"Keys": (uid, iid), "Rating": pred_rating})
            df = pd.concat([df, new_row.to_frame().T])
            i += 1
        # 将预测结果保存到CSV文件
        df.to_csv('submission.csv', index=False)


# 接着是加载测试集并使用LFM模型进行评分预测：
# 定义列数据类型并加载测试集
test_dataset = pd.read_csv('test.csv', dtype=dict(dtype))
# 创建LFM模型实例，设置学习率，正则化系数，隐因子数，迭代次数和列名
lfm = LFM(0.02, 0.01, 0.01, 10, 50, ["userId", "productId", "Rating"])
# 使用训练集数据训练模型
lfm.fit(train_dataset)
# 使用测试集数据测试模型，生成评分预测
lfm.test(test_dataset)

