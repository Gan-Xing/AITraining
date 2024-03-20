import pandas as pd
from autogluon.tabular import TabularPredictor

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 定义预测目标列名
label = 'Rating'

# 删除目标列中的空值行
train_data = train_data.dropna(subset=[label])


# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# AutoGluon训练，加入多模态配置
predictor = TabularPredictor(label=label,eval_metric='root_mean_squared_error').fit(
    train_data,
    presets=['high_quality'] 
)

# 预测测试集
predictions = predictor.predict(test_data)

# 把预测结果作为新列添加到测试数据的副本中
test_data_with_predictions = test_data.copy()
test_data_with_predictions[label] = predictions

# 创建Keys列，直接将userId和productId组合成一个元组
test_data_with_predictions['Keys'] = list(zip(test_data_with_predictions['userId'], test_data_with_predictions['productId']))

# 仅保留Keys和预测结果列
submission = test_data_with_predictions[['Keys', label]]

# 保存预测结果到CSV文件，以符合提交格式
submission.to_csv('submission.csv', index=False)

