from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston = load_boston()



print(boston.keys())

print(boston.data.shape)

print(boston.feature_names)

data = pd.DataFrame(boston.data)
# 赋值列名
data.columns = boston.feature_names

# 目标值
data['PRICE'] = boston.target

# print(data.describe())

X, y = data.iloc[:, :-1], data.iloc[:, -1]

# 将数据结构转为xgboost可支持的数据结构
data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 下一步是通过XGBRegressor()从XGBoost库调用类并将超参数作为参数传递来实例化XGBoost回归对象。
# 对于分类问题，您可以使用XGBClassifier()该类。
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)

# 适合回归训练集，并使用熟悉的测试集的预测.fit()和.predict()方法。
xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

# 通过调用mean_sqaured_errorsklearn metrics模块中的函数来计算rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# 进行k-fold交叉验证
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print("--------------------cv_result----------------------")
cv_results.head()