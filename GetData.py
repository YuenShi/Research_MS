# Author： 施源 Kris
# Create Time： 2018.8.15

import numpy as np
import pandas as pd
import quandl  # 用quandl的api获取stock price 以苹果公司股票进行实验 代码-APPL

API_KEY = "gG8vr-_3fVigtYzrQf5B" # 使用代码者需替换为自己的api_key
quandl.ApiConfig.api_key = API_KEY


def get_data(symbol,n_samples,save_to_csv=False):

    # 股票名称 —— 用户输入
    _symbol = symbol #默认为AAPL

    # 取最近N条数据 —— 用户输入
    _n_samples = n_samples #默认为3000

    # 拼凑用于请求的参数
    request_stock = 'WIKI/' + _symbol
    # print(request_stock)

    # 请求数据
    # TODO：需处理网络异常请求
    try:
        data = quandl.Dataset(request_stock).data()
        # 转换为pandas的Dataframe形式
        df = data.to_pandas()
        # 保存数据到本地csv文件
        if save_to_csv:
            data.to_csv( symbol + '_data.csv')
         # 过老的数据没有太大的参考价值，取最近的n_samples条数据 
        _df = df[-n_samples:]
        return _df
    except IOError as e:
        print(e)
        return None
        