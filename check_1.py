# -*- coding: utf-8 -*-
# Анализ 1 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
#from utils import transform_datetime_features

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_1_r')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))
print('Train dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))

# Данные для провекри
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df.shape))
print('Test dataset memory usage {:.3} MB'.format(df_test.memory_usage().sum() / 1024 / 1024))
# Данные с правильными ответами
y_true = pd.read_csv(target_csv)
#y_true[y_true['target'] > 0] # 172 115 0.668
#

# Для классификации составляем датасет из признаков есть target или нет

y_true['target'] = (y_true['target'] > 0).astype(np.int8)
y_true['target'].hist(bins=100)
df['target'].hist(bins=100)

df_y = df.target
df_X = df.drop('target', axis=1)
# Удаление line_id
df_X = df_X.drop('line_id', axis=1)

# Для регрессии оставляем только определенные значения target (> 0)
df_X = df[df['target'] > 0].copy()
df_y = df_X['target'].copy()
df_X = df_X.drop('target', axis=1)

df_test = pd.merge(df_test, y_true, on='line_id')
df_test = df_test[df_test['target'] > 0]
y_true = df_test[['target']]
df_test = df_test.drop('target', axis=1)

# В начале классифицируем по признаку надо ли делать регресию или нет, target > 0


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


def prepare(df):
    # преобразование колонок с малым количеством уникальных знчений в категории

    for col_name in df.columns:
        if col_name.startswith('number'):
            if df[col_name].unique().shape[0] < 32:
                df[col_name] = df[col_name].astype('category').cat.codes
    return df

df_X = prepare(df_X)

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
# 5 несколько выделенных центров
# 10, 18, 19 почти пустая

plt.scatter(df_X.iloc[:, 1], df_y)
plt.xlabel('x')
plt.ylabel('target')

plt.plot(x=df_X.iloc[:, 6].values, y=df_y.values)

used_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X[used_columns].values

X_test = prepare(df_test[used_columns].copy()).values
X_test = df_test[used_columns].values

model = LogisticRegression()

model = Ridge()
model = LGBMRegressor(n_estimators=50)
model.fit(X_values, df_y)
prediction = model.predict(X_test)
y_true['prediction'] = prediction

metric = mean_squared_error(y_true['target'], y_true['prediction'])
print('RMSE: {:.4}'.format(metric))
# Отправлено 187.1521
# 130.0
# 135.1
# 89.87

y_true['prediction'].hist(bins=100)

#%% Важность признаков
fi = pd.DataFrame(list(zip(df_X.columns[2:], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
