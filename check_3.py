# -*- coding: utf-8 -*-
# Анализ 3 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from utils import transform_datetime_features

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_3_r')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')


df = pd.read_csv(train_csv)
print('Dataset read, shape {}'.format(df.shape))
print('Dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))
df_test = pd.read_csv(test_csv)
print('Test Dataset read, shape {}'.format(df.shape))
y_true = pd.read_csv(target_csv)
y_true = y_true[df_test['id_0'] == 101]

df_y = df.target
df_X = df.drop('target', axis=1)
df_X = df.copy()

df_X = transform_datetime_features(df_X)
df_test = transform_datetime_features(df_test)

df_X.head()
df_X.describe()

df_X['id_0'].value_counts()

df_X['id_0'].unique()
df_u = df_X[df_X['id_0'] == 102]
#df_u[['datetime_0', 'target']].plot()
df_y = df_X[df_X['id_0'] == 102].target

df_u_test = df_test[df_test['id_0'] == 102]

df_u['target'].plot()
y_true['target'].plot()

# Удаление колонок с постоянными значениями
constant_columns = [
    col_name
    for col_name in df_X.columns
    if df_X[col_name].nunique() == 1
]
df_X.drop(constant_columns, axis=1, inplace=True)

ONEHOT_MAX_UNIQUE_VALUES = 20
def prepare(df):
    categorical_values = {}
    for col_name in list(df.columns):
        col_unique_values = df[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    return df

df_X = prepare(df_X)
df_test = prepare(df_test)

# Корреляция
correlations = np.abs([
    np.corrcoef(df_y, df_X[col_name])[0, 1]
    for col_name in df_X.columns if col_name.startswith('number')
])
np.sort(correlations)

# Посмотрим на данные
[ df_X[col].unique().shape[0] for col in df_X.columns]


df_X.iloc[:, 15].unique().shape[0]
df_X.iloc[:, 15].value_counts().sort_values()
df_X.iloc[:, 41].value_counts().sort_values()
# 6, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 - категория

# Посмотрим на зависимость целевой переменной от остальных
#

plt.scatter(df_u.iloc[:, 33], df_u['target'])
plt.xlabel('x')
plt.ylabel('target')

plt.plot(x=df_X.iloc[:, 6].values, y=df_y.values)

used_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_u[used_columns].fillna(-1).values
X_values = df_X.values

X_test = prepare(df_test[used_columns].copy()).values
X_test = prepare(df_test.copy()).values
X_test = df_u_test[used_columns].fillna(-1).values

model = LGBMRegressor(n_estimators=100)
model = Ridge()
model.fit(X_values, df_y)
prediction = model.predict(X_test)
result = y_true.copy()
result['prediction'] = prediction

metric = mean_squared_error(result['target'], result['prediction'])
print('RMSE: {}'.format(metric))
# Отправлено
#1210754018
#1204794882

result['prediction'].plot()

#%% Важность признаков
fi = pd.DataFrame(list(zip(df_X.columns[2:], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
