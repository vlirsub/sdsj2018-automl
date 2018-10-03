# -*- coding: utf-8 -*-
# Анализ 6 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
# from utils import transform_datetime_features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from utils import transform_datetime_features

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_6_c')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))
print('Train dataset memory usage {} MB'.format(df.memory_usage().sum() / 1024 / 1024))

# Данные для провекри
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df.shape))
print('Test dataset memory usage {:.3} MB'.format(df_test.memory_usage().sum() / 1024 / 1024))
# Данные с правильными ответами
y_true = pd.read_csv(target_csv)

# Для классификации составляем датасет из признаков есть target или нет
df_X = df.copy()
df_X.shape  # 467485, 17

df_y = df_X[['target']].copy()
df_y.shape  # 467485, 1
df_X = df_X.drop('target', axis=1)
df_X.shape  # 467485, 16
# Правильные ответы
y_test = y_true.copy()
y_test.shape  # 169638, 2

# drop constant features
constant_columns = [
    col_name
    for col_name in df_X.columns
    if df_X[col_name].nunique() == 1
]
df_X.drop(constant_columns, axis=1, inplace=True)

# features from datetime
df_X = transform_datetime_features(df_X)
df_test = transform_datetime_features(df_test)

# df_X['datetime_0'].hist()
# df_X['datetime_0'].unique()
# df_X['datetime_0'].value_counts()
# df_test['datetime_0'].value_counts()

ONEHOT_MAX_UNIQUE_VALUES = 20

categorical_values = {}
for col_name in list(df_X.columns):
    col_unique_values = df_X[col_name].unique()
    if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
        categorical_values[col_name] = col_unique_values
        for unique_value in col_unique_values:
            df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)

# missing values
if any(df_X.isnull()):
    missing = True
    df_X.fillna(-1, inplace=True)

# for col_name in df.columns:
#     if col_name.startswith('number') or col_name.startswith('string'):
#         if df[col_name].unique().shape[0] < 32:
#             df[col_name] = df[col_name].astype('category').cat.codes


# categorical encoding
for col_name, unique_values in categorical_values.items():
    for unique_value in unique_values:
        df_test['onehot_{}={}'.format(col_name, unique_value)] = (df_test[col_name] == unique_value).astype(int)

# missing values
if missing:
    df_test.fillna(-1, inplace=True)
elif any(df_test.isnull()):
    df_test.fillna(value=df_test.mean(axis=0), inplace=True)

used_columns = [
    col_name
    for col_name in df_X.columns
    if col_name.startswith('number') or col_name.startswith('onehot') or col_name.startswith('dt') #
]
X_values = df_X[used_columns].values
y_train = df_y.values.ravel()
X_test = df_test[used_columns].values
print('X_values shape {}'.format(X_values.shape))  # X_values shape (467485, 89)
print('X_test shape {}'.format(X_test.shape))  # X_test shape (169638, 89)

# PolynomialFeatures
pf = PolynomialFeatures()
X_values = pf.fit_transform(X_values)
X_test = pf.transform(X_test)

model = LGBMClassifier(n_estimators=100)
model.fit(X_values, y_train)
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
score = cross_val_score(model, X_values, y_train, cv=kfold, n_jobs=1, scoring='roc_auc',
                        verbose=0)

print('score {:.4}'.format(score.mean()))
#score 0.7853 roc auc: 0.783 col_name.startswith('number') or col_name.startswith('dt') or col_name.startswith('onehot')
# score 0.781 roc auc: 0.7787 if col_name.startswith('number')
# score 0.7832  roc auc: 0.7809  col_name.startswith('number') or col_name.startswith('onehot')
# score 0.7854 roc auc: 0.7831 col_name.startswith('number') or col_name.startswith('dt')

prediction = model.predict_proba(X_test)[:, 1]

result = y_true.copy()
result['prediction'] = prediction

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.4}'.format(metric))
# Отправлено

result['prediction'].hist(bins=100)

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X[used_columns], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
