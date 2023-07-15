import pandas as pd
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def BP(data: pd.DataFrame):
    # 分割特征数据和性质数据
    features = data.drop(['大豆油浓度', '玉米油浓度', '茶油浓度', 'id'], axis=1)
    labels = data[['大豆油浓度', '玉米油浓度', '茶油浓度']]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 创建BP神经网络回归模型
    mlp = MLPRegressor(random_state=42)

    layerIndex = (100, 200, 400, 600, 800)
    layerIndex_re = (800, 600, 400, 200, 100)
    layerIndex_random = (123, 452, 115, 981, 215)

    # 定义参数网格
    param_grid = {
        "hidden_layer_sizes": [layerIndex, layerIndex_re, layerIndex_random],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # 网格搜索获取最优参数
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 打印最优参数和得分
    print("最优参数:", grid_search.best_params_)
    print("最优得分:", -grid_search.best_score_)

    # 使用最优参数的模型进行预测
    mlp_best = grid_search.best_estimator_
    y_pred_best = mlp_best.predict(X_test)

    # 计算最优参数下的评估指标
    rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
    r2_best = r2_score(y_test, y_pred_best)
    print("最优参数下的RMSE:", rmse_best)
    print("最优参数下的R方指标:", r2_best)

    # 使用默认参数的模型进行预测
    mlp_default = MLPRegressor(random_state=42)
    mlp_default.fit(X_train, y_train)
    y_pred_default = mlp_default.predict(X_test)

    # 计算默认参数下的评估指标
    rmse_default = mean_squared_error(y_test, y_pred_default, squared=False)
    r2_default = r2_score(y_test, y_pred_default)
    print("默认参数下的RMSE:", rmse_default)
    print("默认参数下的R方指标:", r2_default)


if __name__ == '__main__':
    data = pd.read_csv('../../实验数据、论文/数据/双掺样本.csv')
    BP(data)