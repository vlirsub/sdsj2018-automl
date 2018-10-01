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
df_X_c = df.copy()
df_X_c.shape # 365, 42
df_X_c['target'] = (df_X_c['target'] > 0).astype(np.int8)

df_y_c = df_X_c[['target']].copy()
df_y_c.shape # 365, 1
df_X_c = df_X_c.drop('target', axis=1)
df_X_c.shape # 365, 41
# Правильные ответы
y_true_c = y_true.copy()
y_true_c['target'] = (y_true_c['target'] > 0).astype(np.int8)
y_true_c.shape # 172, 2

# Для регрессии оставляем только заданные target
df_X_r = df[df['target'] > 0].copy()
df_y_r = df_X_r[['target']].copy()
df_X_r = df_X_r.drop('target', axis=1)
df_X_r.shape # 246, 41
df_y_r.shape # 246, 1
# Правильные ответы
y_true_r = y_true[y_true['target'] > 0].copy()
y_true_r.shape # 115, 2

# В начале классифицируем по признаку надо ли делать регресию или нет, target > 0
used_columns = [
    col_name
    for col_name in df_X_c.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X_c[used_columns].values
X_test = df_test[used_columns].values
print('X_values shape {}'.format(X_values.shape)) # X_values shape (365, 39)
print('X_test shape {}'.format(X_test.shape)) # X_test shape (172, 39)

model = LogisticRegression()
model.fit(X_values, df_y_c['target'])
# Проноз
prediction = model.predict(X_test)
# Результат классификации
result_c = y_true_c.copy()
result_c['prediction'] = prediction

metric = roc_auc_score(result_c['target'], result_c['prediction'])
print('roc auc: {:.4}'.format(metric))

# Теперь из тестовых данных надо классифицировать только те у которых target > 0
X_values = df_X_r[used_columns].values
X_test = df_test[used_columns][result_c['prediction'] > 0].values
print('X_values shape {}'.format(X_values.shape)) # X_values shape (246, 39)
print('X_test shape {}'.format(X_test.shape)) # X_test shape (115, 39)


model = Ridge()
model = LGBMRegressor(n_estimators=50)
model.fit(X_values, df_y_r['target'])
# Проноз
prediction = model.predict(X_test)
# Результат регрессии
result_r = y_true_r.copy()
result_r['prediction'] = prediction

metric = mean_squared_error(y_true['target'], y_true['prediction'])
print('RMSE: {:.4}'.format(metric))


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
