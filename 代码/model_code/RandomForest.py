import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def RandomForest(data: pd.DataFrame):
    # 分割特征数据和性质数据
    features = data.drop(['大豆油浓度', '玉米油浓度', '茶油浓度', 'id'], axis=1)
    labels = data[['大豆油浓度', '玉米油浓度', '茶油浓度']]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 创建随机森林回归模型
    rf = RandomForestRegressor(random_state=42)

    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 15, 30, 60],
        'min_samples_split': [2, 4, 6, 10, 15],
        'min_samples_leaf': [1, 3, 5, 9]
    }

    # 网格搜索获取最优参数
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    # 打印最优参数和得分
    print("最优参数:", grid_search.best_params_)
    print("最优得分:", -grid_search.best_score_)

    # 使用最优参数的模型进行预测
    rf_best = grid_search.best_estimator_
    y_pred_best = rf_best.predict(X_test)

    # 计算最优参数下的评估指标
    rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
    r2_best = r2_score(y_test, y_pred_best)
    print("最优参数下的RMSE:", rmse_best)
    print("最优参数下的R方指标:", r2_best)

    # 使用默认参数的模型进行预测
    rf_default = RandomForestRegressor(random_state=42)
    rf_default.fit(X_train, y_train)
    y_pred_default = rf_default.predict(X_test)

    # 计算默认参数下的评估指标
    rmse_default = mean_squared_error(y_test, y_pred_default, squared=False)
    r2_default = r2_score(y_test, y_pred_default)
    print("默认参数下的RMSE:", rmse_default)
    print("默认参数下的R方指标:", r2_default)


if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('../../实验数据、论文/数据/双掺样本.csv')
    RandomForest(data)