# -*- coding: utf-8 -*-
# Анализ 7 набора данных

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, mean_squared_error
# from utils import transform_datetime_features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from utils import transform_datetime_features

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_7_c')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))  # (92091, 774)
print('Train dataset memory usage {} MB'.format(df.memory_usage().sum() / 1024 / 1024))  # 543.8113708496094 MB

# Данные для провекри
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df_test.shape))  # (87062, 773)
print('Test dataset memory usage {:} MB'.format(df_test.memory_usage().sum() / 1024 / 1024))  # 513.4501342773438 MB
# Данные с правильными ответами
y_true = pd.read_csv(target_csv)

# Для классификации составляем датасет из признаков есть target или нет
df_X = df.copy()
df_X.shape  # (92091, 774)

df_y = df_X[['target']].copy()
df_y.shape  # (92091, 1)
df_X = df_X.drop('target', axis=1)
df_X.shape  # 467485, 16
# Правильные ответы
y_test = y_true.copy()
y_test.shape  # 169638, 2

df_X = transform_datetime_features(df_X)
# drop constant features
constant_columns = [
    col_name
    for col_name in df_X.columns
    if df_X[col_name].nunique() == 1
]
df_X.drop(constant_columns, axis=1, inplace=True)

# Анализ данных
#
# number — числовой формат (может содержать количественную, категориальную или бинарную величину)
# string — строковый формат
# datetime — дата в формате 2010-01-01 или дата/время в формате 2010-01-01 10:10:10
# id — идентификатор (категориальная величина особой природы)

number_columns = [col_name for col_name in df_X.columns if col_name.startswith('number')]
string_columns = [col_name for col_name in df_X.columns if col_name.startswith('string')]
datetime_columns = [col_name for col_name in df_X.columns if col_name.startswith('datetime')]
id_columns = [col_name for col_name in df_X.columns if col_name.startswith('id')]

for sc in string_columns:
    print('{} {}'.format(sc, df_X[sc].unique()))
# string_0 [nan 'DSA' 'Other']
# string_1 ['living in city in apart' 'other']
# string_2 ['N' 'Y' nan]

# ['datetime_0', 'datetime_1']
for с in datetime_columns:
    print('{} {}'.format(с, df_X[с].unique()))

# ['id_0']
for с in id_columns:
    print('{} {}'.format(с, df_X[с].unique()))

df_X['target'].value_counts()
# 0.0    91121
# 1.0      970

y_true['target'].value_counts()
# 0.0    86469
# 1.0      593


df_X['datetime_0'].hist()
df_X['datetime_1'].hist()
df_X['id_0'].hist()
df_X['number_24'].hist(bins=100)


df_X['datetime_0'] = pd.to_datetime(df_X['datetime_0'])
df_X['datetime_1'] = pd.to_datetime(df_X['datetime_1'])
df_test['datetime_0'] = pd.to_datetime(df_test['datetime_0'])
df_test['datetime_1'] = pd.to_datetime(df_test['datetime_1'])

df_X['number_{}_{}'.format('datetime_0', 'datetime_1')] = (df_X['datetime_0'] - df_X['datetime_1']).dt.days
df_test['number_{}_{}'.format('datetime_0', 'datetime_1')] = (df_test['datetime_0'] - df_test['datetime_1']).dt.days


# Колонки с шумом
def noise_columns(df, val):
    u = df.shape[0]
    return [col_name for col_name in df.columns if df[col_name].unique().shape[0] / u >= val]


noise_columns(df_X, 0.97)
# ['id_0', 'line_id']

# Корреляция
correlations = np.abs([
    np.corrcoef(df_y['target'], df_X[col_name].fillna(-1).values)[0, 1]
    for col_name in df_X.columns if col_name.startswith('number')
])
np.sort(correlations)[-10:]

# Количество уникальных значений
[(col_name, df_X[col_name].unique().shape[0]) for col_name in df_X.columns]

multi_columns = [col_name for col_name in df_X.columns if df_X[col_name].unique().shape[0] > 2]

# features from datetime
df_X = transform_datetime_features(df_X)
df_test = transform_datetime_features(df_test)

# df_X['datetime_0'].hist()
# df_X['datetime_0'].unique()
# df_X['datetime_1'].value_counts()
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
    if col_name.startswith('number') or col_name.startswith('onehot') or col_name.startswith('dt')  #
]
X_values = df_X[used_columns].values
y_train = df_y.values.ravel()
X_test = df_test[used_columns].values
print('X_values shape {}'.format(X_values.shape))  # X_values shape (92091, 241)
print('X_test shape {}'.format(X_test.shape))  # X_test shape (87062, 241)

# PolynomialFeatures
pf = PolynomialFeatures()
X_values = pf.fit_transform(X_values)
X_test = pf.transform(X_test)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    # 'objective': 'regression' if args.mode == 'regression' else 'binary',
    'objective': 'binary',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 200,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.70,
    'bagging_freq': 4,
    "max_depth": -1,
    "verbosity": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    # "min_split_gain":0.2,
    "min_child_weight": 10,
    'zero_as_missing': True,
    'num_threads': 8,
}

model = lgb.train(params, lgb.Dataset(X_values, label=y_train), 600)

model = LGBMClassifier(n_estimators=100)
model.fit(X_values, y_train)
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
score = cross_val_score(model, X_values, y_train, cv=kfold, n_jobs=1, scoring='roc_auc', verbose=0)
print('score {:.4}'.format(score.mean()))

prediction = model.predict_proba(X_test)[:, 1]
prediction = model.predict(X_test)

result = y_true.copy()
result['prediction'] = prediction

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.4}'.format(metric))
# 0.8453
# 0.8317

result['prediction'].hist(bins=100)

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X[used_columns], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
