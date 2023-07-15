import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def XGBoost(data: pd.DataFrame):
    # 分割特征数据和性质数据
    features = data.drop(['大豆油浓度', '玉米油浓度', '茶油浓度', 'id'], axis=1)
    labels = data[['大豆油浓度', '玉米油浓度', '茶油浓度']]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 将数据转换为DMatrix格式
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置XGBoost参数范围
    param_grid = {
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [1, 0.1, 0.01, 0.001],
        'n_estimators': [50, 100, 500, 1000, 2000, 5000]
    }

    # 创建XGBoost回归模型z
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # 创建网格搜索对象
    grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)

    # 在训练集上进行超参数搜索
    grid_search.fit(X_train, y_train)

    # 输出最优的超参数组合
    best_params = grid_search.best_params_
    print("最优超参数组合：", best_params)

    # 使用最优超参数训练模型
    best_model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
    best_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)

    # 计算均方根误差
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('均方根误差（RMSE）:', rmse)

    # 计算R方指标
    r2 = r2_score(y_test, y_pred)
    print('R方指标:', r2)

    print("默认超参数：")
    # 使用默认超参数训练模型

    base_model = xgb.XGBRegressor(objective='reg:squarederror')
    base_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = base_model.predict(X_test)

    # 计算均方根误差
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('均方根误差（RMSE）:', rmse)

    # 计算R方指标
    r2 = r2_score(y_test, y_pred)
    print('R方指标:', r2)


if __name__ == '__main__':

    data = pd.read_csv('../../实验数据、论文/数据/双掺样本.csv')

    # 调用XGBoost函数进行模型训练和评估
    XGBoost(data)
