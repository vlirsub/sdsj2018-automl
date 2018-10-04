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

df_y = df.target
df_X = df.drop('target', axis=1)
df_X = df.copy()

#df_X = transform_datetime_features(df_X)
#df_test = transform_datetime_features(df_test)

#df_X.head()
#df_X.describe()

number_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number')
]

def f_trans(x):
    for cn in number_columns:
        for i in range(1, 2 * 7 + 1):
            x['{}_s{}'.format(cn, i)] = x[cn].shift(i)
    return x

df_X_s = df_X[['id_0'] + number_columns].groupby('id_0').apply(f_trans)
df_test_s = df_test[['id_0'] + number_columns].groupby('id_0').apply(f_trans)

# Проверяем
#df_X_s[df_X_s['id_0'] == 500]
#df_test_s[df_test_s['id_0'] == 500]

# Удаление колонок с постоянными значениями
constant_columns = [
    col_name
    for col_name in df_X_s.columns
    if df_X_s[col_name].nunique() == 1
]
df_X_s.drop(constant_columns, axis=1, inplace=True)

ONEHOT_MAX_UNIQUE_VALUES = 32
def prepare(df):
    categorical_values = {}
    for col_name in list(df.columns):
        col_unique_values = df[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    return df

#df_X = prepare(df_X)
#df_test = prepare(df_test)

# Корреляция
correlations = np.abs([
    np.corrcoef(df_y, df_X_s[col_name].fillna(-1).values)[0, 1]
    for col_name in df_X_s.columns if col_name.startswith('number')
])
np.sort(correlations)[-10:]

# Посмотрим на данные
#[ df_X[col].unique().shape[0] for col in df_X.columns]

# Используемые для обучения колонки
used_columns = [
    col_name
    for col_name in df_X_s.columns
    if col_name.startswith('number') or col_name.startswith('onehot') or col_name.startswith('dt')
]
X_values = df_X_s[used_columns].fillna(-1).values
#X_values = df_X.values

#X_test = prepare(df_test[used_columns].copy()).values
#X_test = prepare(df_test.copy()).values
X_test = df_test_s[used_columns].fillna(-1).values

model = LGBMRegressor(n_estimators=200)
#model = Ridge(normalize=True)
model.fit(X_values, df_y)
prediction = model.predict(X_test)
result = y_true.copy()
result['prediction'] = prediction

metric = mean_squared_error(result['target'], result['prediction'])
print('RMSE: {}'.format(metric))
# Отправлено
#1748460185
#11925728503 Отправлено
#12067769616
#11555808147
#11405518348
#11385751169 200

#%% Важность признаков
fi = pd.DataFrame(list(zip(used_columns, model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
