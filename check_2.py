# -*- coding: utf-8 -*-
# Анализ 2 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from utils import transform_datetime_features
from sklearn.preprocessing import StandardScaler

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_2_r')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))
print('Train dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df.shape))
# Правильные ответы
y_true = pd.read_csv(target_csv)

df_y = df.target
df_X = df.drop('target', axis=1)
# df_X = df.copy()
# Удаление line_id
df_X = df_X.drop('line_id', axis=1)
df_X = df_X.drop('is_test', axis=1)

df_test = df_test.drop('line_id', axis=1)
df_test = df_test.drop('is_test', axis=1)

np.sort(df_X[['string_0', 'string_1']].apply(lambda x: '{}_{}'.format(x[0], x[1]), axis=1).unique())
np.sort(df_test[['string_0', 'string_1']].apply(lambda x: '{}_{}'.format(x[0], x[1]), axis=1).unique())

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
    # categorical_values = {}
    # for col_name in list(df.columns):
    #     col_unique_values = df[col_name].unique()
    #     if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
    #         categorical_values[col_name] = col_unique_values
    #         for unique_value in col_unique_values:
    #             df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    #
    for col_name in df.columns:
        if col_name.startswith('number') or col_name.startswith('string'):
            if df[col_name].unique().shape[0] < 32:
                df[col_name] = df[col_name].astype('category').cat.codes

    return df


df_X = prepare(df_X)
df_test = prepare(df_test)

df_X = df_X.fillna(-1)
df_test = df_test.fillna(-1)

d = df_X.groupby(['string_0', 'string_1']).mean().to_dict()


def f_app(x):
    df = pd.DataFrame()
    if x.shape[0] > 0:
        k1 = x['string_0'].iloc[0]
        k2 = x['string_1'].iloc[0]
        # print(k1, k2)
        # d['number_0'][0, 2]
        # print(x.columns)
        for col_name in x.columns:
            if col_name.startswith('number'):
                df[col_name] = x[col_name] - d[col_name][k1, k2]
    return df


df_X_mean = df_X.groupby(['string_0', 'string_1']).apply(f_app).add_suffix('_dist_01_mean')
df_X_mean = df_X_mean.reset_index()
df_X_mean = df_X_mean.drop('level_2', axis=1)
df_X_mean['string_0'] = df_X_mean['string_0'].astype(np.int8)
df_X_mean['string_1'] = df_X_mean['string_1'].astype(np.int8)

# df_X_mean = df_X.groupby(['string_0', 'string_1']).transform(lambda x : x - x.mean()).add_suffix('_dist_01_mean')
df1 = df_X[(df_X['string_0'] == 6) & (df_X['string_1'] == 0)][['string_0', 'string_1', 'number_0']].head().copy()
df2 = df_X[(df_X['string_0'] == 6) & (df_X['string_1'] == 0)][['string_0', 'string_1', 'number_0']].head().copy()

pd.merge(df1, df2, on=['string_0', 'string_1'], how='left')
df1.merge(df2, on=['string_0', 'string_1'], how='left')

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

result = pd.merge(left, left, on=['key', 'A'], how='left')

df_X = pd.merge(df_X[(df_X['string_0'] == 6) & (df_X['string_1'] == 0)][['string_0', 'string_1', 'number_0']],
                df_X_mean[(df_X_mean['string_0'] == 6) & (df_X_mean['string_1'] == 0)][
                    ['string_0', 'string_1', 'number_0_dist_01_mean']],
                on=['string_0', 'string_1'], how='left')
df_X[['string_0', 'string_1']]
df_X_mean[['string_0', 'string_1']]
df_X['string_0']
df_X_mean['string_0']

df_test = df_test[~((df_test['string_0'] == 4) & (df_test['string_1'] == 2))]

df_test_mean = df_test.groupby(['string_0', 'string_1']).apply(f_app)
df_test_mean = df_test_mean.reset_index()
df_test_mean = df_test_mean.drop('level_2', axis=1)

df_test = pd.merge(df_test, df_test_mean, on=['string_0', 'string_1'])

# df_X_mean = df_X.groupby(['string_0', 'string_1']).transform(lambda x : x - x.mean()).add_suffix('_dist_01_mean')
df_X = pd.merge(df_X, df_X_mean, on=['string_0', 'string_1'])

# Корреляция
correlations = np.abs([
    np.corrcoef(df_y, df_X[col_name])[0, 1]
    for col_name in df_X.columns if col_name.startswith('number')
])
np.sort(correlations)

# Посмотрим на данные
df_X.iloc[:, 0].value_counts().sort_values()
df_X.iloc[:, 41].value_counts().sort_values()
# 6, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 - категория

# Посмотрим на зависимость целевой переменной от остальных
# 2 3 4 5 - log?

plt.scatter(df_X['number_0'], df_y)
np.corrcoef(df_X['number_0'].fillna(1), df_y)

df_y[df_y.isna()]
l = np.log(df_X['number_0'].fillna(1))
l[~np.isfinite(l)] = 0
np.corrcoef(l, df_y)

plt.scatter(l, df_y)
plt.xlabel('x')
plt.ylabel('target')

plt.plot(x=df_X.iloc[:, 6].values, y=df_y.values)

used_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X[used_columns].values
X_values = df_X.fillna(-1).values

X_test = prepare(df_test[used_columns].copy()).values
X_test = df_test[used_columns].values
X_test = df_test.fillna(-1).values

scaler = StandardScaler()
X_values = scaler.fit_transform(X_values)
X_test = scaler.transform(X_test)

model = Ridge()
model = LGBMRegressor(n_estimators=100)
model.fit(X_values, df_y)
prediction = model.predict(X_test)
y_true['prediction'] = prediction

metric = mean_squared_error(y_true['target'], y_true['prediction'])
print('RMSE: {:.4}'.format(metric))
# Отправлено 1.6500
# 1.575
# 1.573

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X.columns[2:], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
