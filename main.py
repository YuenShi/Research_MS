# Author： 施源 Kris
# Create Time： 2018.8.15

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入项目所需的模块文件
import GetData
import DataPreprocess
import StockEnvironment
import DQN
import Runner


# 用户输入内容，已赋默认值
symbol = 'AAPL'  # 股票名称
n_samples = 800  # 数据量
split_ratio = 0.5  # 训练测试集划分比例

# 全局变量
N_FEATURES = 8   # 每个state的特征数量（open、high、low、close、volumn、tradePrice、cash、stockValue）

N_TIMESTEPS = 20 # 每个state的步长

N_ACTIONS =  3 # action的可选数量 （hold、buy、sell）

EPSILON = 0.95   # 贪婪度

LEARNING_RATE = 0.001     # 学习率

GAMMA = 0.9    # 奖励折扣率 

EPOCHS = 3000   # 训练回合数

BATCH_SIZE = 64  # 批处理样本数

ORDER_SIZE = 10 # 每次下单交易股数

REPLACE_TARGET_ITER = 200  # DQN网络更换target_net的步数

MEMORY_SIZE = 2000 # 记忆上限



def main():
	# 从Quandl获取数据
	data = GetData.get_data(symbol, n_samples)
	# 测试用本地数据
	# data = pd.read_csv('data.csv')
	# 数据预处理 返回划分好的训练集和测试集
	train, test = DataPreprocess.data_preprocess(data, split_ratio)

	# 生成训练环境和测试环境
	env_train = StockEnvironment.StockEnv(train)
	env_test = StockEnvironment.StockEnv(test)

	N_ACTIONS = len(env_train.actions)
	N_FEATURES = env_train.states.shape[1]

	# 初始化runner
	runner = Runner.Runner()
	# 训练dqn网络，返回训练完毕的模型，以及训练最终结果; 显示训练情况图
	trained_dqn = runner.trainer(env_train, 20)

	# 用训练后的trained_Q对test数据进行分析，给出预测出的最终交易行为；显示测试情况图
	act = runner.tester(env_test, trained_dqn)
	# 预测说明：
	#          模型仅预测当天的交易行为，输入模型的数据为历史数据，
	#		   但不是输入train数据集的数据，而是test数据集的数据



if __name__ == '__main__':
    main()




