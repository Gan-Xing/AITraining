import pandas as pd
from autogluon.tabular import TabularPredictor

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 删除Summary列，因为文本太多特征提取与分析耗时太长收益不高
# 删除Summary列和Id列
train_data = train_data.drop(columns=['Id'])

# 定义预测目标列名
label = 'SalePrice'

# 删除目标列中的空值行
train_data = train_data.dropna(subset=[label])


# 加载数据
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# 删除Summary列，因为文本太多特征提取与分析耗时太长收益不高
# 删除Summary列和Id列
train_data = train_data.drop(columns=['Id'])

# 定义预测目标列名
label = 'SalePrice'

# 删除目标列中的空值行
train_data = train_data.dropna(subset=[label])
# AutoGluon训练，加入多模态配置
predictor = TabularPredictor(label=label).fit(
    train_data,
    presets=['best_quality'] 
)
  # presets='best_quality',  # 使用最佳质量预设
    # hyperparameters='multimodal',  # 使用多模态配置，适合有GPU时使用
# 预测测试集
predictions = predictor.predict(test_data)

# 把预测结果作为新列添加到测试数据的副本中
test_data_with_predictions = test_data.copy()
test_data_with_predictions[label] = predictions

# 仅保留Id和预测结果列
submission = test_data_with_predictions[['Id', label]]

# 保存预测结果到CSV文件，以符合提交格式
submission.to_csv('/kaggle/working/final_predictions.csv', index=False)
