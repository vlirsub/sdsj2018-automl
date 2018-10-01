# -*- coding: utf-8 -*-
# Анализ 3 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Dataset read, shape {}'.format(df.shape))
print('Dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))
df_test = pd.read_csv(test_csv)
print('Test Dataset read, shape {}'.format(df.shape))
y_true = pd.read_csv(target_csv)

df_y = df.target
df_X = df.drop('target', axis=1)
# Удаление line_id
df_X = df_X.drop('line_id', axis=1)

df_test = df_test.drop('line_id', axis=1)

df_X = transform_datetime_features(df_X)
df_test = transform_datetime_features(df_test)

df_X.head()
df_X.describe()

# Удаление колонок с постоянными значениями
constant_columns = [
    col_name
    for col_name in df_X.columns
    if df_X[col_name].nunique() == 1
]
df_X.drop(constant_columns, axis=1, inplace=True)

ONEHOT_MAX_UNIQUE_VALUES = 20
def prepare(df):
    # преобразование колонок с малым количеством уникальных знчений в категории

    for col_name in df.columns:
        if col_name.startswith('number') or col_name.startswith('string') :
            if df[col_name].unique().shape[0] < 32:
                df[col_name] = df[col_name].astype('category').cat.codes

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

plt.scatter(df_X.iloc[:, 0], df_y)
plt.xlabel('x')
plt.ylabel('target')

plt.plot(x=df_X.iloc[:, 6].values, y=df_y.values)

used_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X[used_columns].values
X_values = df_X.values

X_test = prepare(df_test[used_columns].copy()).values
X_test = prepare(df_test.copy()).values

model = LGBMRegressor(n_estimators=100)
model.fit(X_values, df_y)
prediction = model.predict(X_test)
y_true['prediction'] = prediction

metric = mean_squared_error(y_true['target'], y_true['prediction'])
print('RMSE: {:.4}'.format(metric))
# Отправлено 1.6500
#1.575

#%% Важность признаков
fi = pd.DataFrame(list(zip(df_X.columns[2:], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
