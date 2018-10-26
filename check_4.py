# -*- coding: utf-8 -*-
# Анализ 4 набора данных

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

from utils import transform_datetime_features

# Директория с данными
DATA_DIR = r'm:\tmp\SB'

fd = os.path.join(DATA_DIR, 'check_4_c')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))
print('Train dataset memory usage {:n} MB'.format(df.memory_usage().sum() / 1024 / 1024))

# Данные для провекри
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df.shape))
print('Test dataset memory usage {:n} MB'.format(df_test.memory_usage().sum() / 1024 / 1024))
# Данные с правильными ответами
y_true = pd.read_csv(target_csv)

df_test = pd.merge(df_test, y_true, on='line_id')

df_X = df.copy()
df_X.shape  # 114130, 143

df_y = df_X[['target']].copy()
df_y.shape  # 114130, 1
df_X = df_X.drop('target', axis=1)
df_X.shape  # 114130, 142
# Правильные ответы
df_X_test = df_test.copy()

# drop constant features
constant_columns = [
    col_name
    for col_name in df_X.columns
    if df_X[col_name].nunique() == 1
]
df_X.drop(constant_columns, axis=1, inplace=True)
df_X_test.drop(constant_columns, axis=1, inplace=True)


# number_88, number_89, number_90, number_91, number_92, number_93, number_94, number_95, number_96, number_97, number_98, number_99, number_100, number_101, number_102, number_103, number_104, number_105, number_106, number_107, number_108, number_109, number_110, number_111, number_112, number_113, number_114, number_115, number_116, number_117, number_118, number_119, number_120, number_121, number_122, number_123, number_124, number_125, number_126, number_127, number_128, number_129, number_130, number_131, number_132, number_133, number_134, number_135, number_136, number_137

## Коллинеарные признаки
def corr_df(x, corr_val):
    # Creates Correlation Matrix and Instantiates
    corr_matrix = x.corr()
    # corr_matrix = df_X[number_columns].corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            val = item.values[0][0]
            if abs(val) >= corr_val:
                col = item.columns[0]
                row = item.index[0]
                # Prints the correlated feature set and the corr val
                print(col, "|", row, "|", round(val, 2))
                drop_cols.append(col)

    drops = set(drop_cols)

    return drops


droped_columns = corr_df(df_X, 0.7)

# Корреляция
correlations = np.abs([
    np.corrcoef(df_y['target'], df_X[col_name].fillna(-1).values)[0, 1]
    for col_name in df_X.columns if col_name.startswith('number')
])
np.sort(correlations)[-10:]

## Уникалные значения колонок
[(col_name, df_X[col_name].unique().shape[0]) for col_name in df_X.columns]
# [('number_0', 31),
# ('number_1', 31),
# ('number_2', 74264),
# ('number_3', 3512),
# ('number_4', 517),
# ('number_5', 2335),
# ('number_6', 324),
# ('number_7', 6),
# ('number_8', 88),
# ('number_9', 26212),
# ('number_10', 17),
# ('number_11', 102203),
# ('number_12', 15),
# ('number_13', 156),
# ('number_14', 2914),
# ('number_15', 1210),
# ('number_16', 15453),
# ('number_17', 367),
# ('number_18', 4),
# ('number_19', 6),
# ('number_20', 4),
# ('number_21', 6),
# ('number_22', 4),
# ('number_23', 4),
# ('number_24', 4),
# ('number_25', 4),
# ('number_26', 4),
# ('number_27', 6),
# ('number_28', 6),
# ('number_29', 4),
# ('number_30', 4),
# ('number_31', 7),
# ('number_32', 4),
# ('number_33', 6),
# ('number_34', 4),
# ('number_35', 4),
# ('target', 2),
# ('datetime_0', 229),
# ('number_36', 92291),
# ('datetime_1', 213),
# ('datetime_2', 339),
# ('number_37', 107380),
# ('line_id', 114130),
# ('number_38', 114130), Скорее всего шум, колчиество значений равно количеству строк
# ('number_39', 114130),
# ('number_40', 114130),
# ('number_41', 114130),
# ('number_42', 114130),
# ('number_43', 114130),
# ('number_44', 114130),
# ('number_45', 114130),
# ('number_46', 114130),
# ('number_47', 114130),
# ('number_48', 114130),
# ('number_49', 114130),
# ('number_50', 114130),
# ('number_51', 114130),
# ('number_52', 114130),
# ('number_53', 114130),
# ('number_54', 114130),
# ('number_55', 114130),
# ('number_56', 114130),
# ('number_57', 114130),
# ('number_58', 114130),
# ('number_59', 114130),
# ('number_60', 114130),
# ('number_61', 114130),
# ('number_62', 114130),
# ('number_63', 114130),
# ('number_64', 114130),
# ('number_65', 114130),
# ('number_66', 114130),
# ('number_67', 114130),
# ('number_68', 114130),
# ('number_69', 114130),
# ('number_70', 114130),
# ('number_71', 114130),
# ('number_72', 114130),
# ('number_73', 114130),
# ('number_74', 114130),
# ('number_75', 114130),
# ('number_76', 114130),
# ('number_77', 114130),
# ('number_78', 114130),
# ('number_79', 114130),
# ('number_80', 114130),
# ('number_81', 114130),
# ('number_82', 114130),
# ('number_83', 114130),
# ('number_84', 114130),
# ('number_85', 114130),
# ('number_86', 114130),
# ('number_87', 114130)]

df_X['number_85'].hist(bins=100)

df_X['datetime_0'].hist(bins=100)
df_X['datetime_1'].hist(bins=100)
df_X['datetime_2'].hist(bins=100)

np.sort(df_X['datetime_0'].unique())
df_X['datetime_0'].value_counts().sort_index()
df_X['datetime_1'].value_counts().sort_index()
df_X['datetime_2'].value_counts().sort_index()

np.sort(df_X_test['datetime_0'].unique())

df_X_test['datetime_0'].value_counts().sort_index()
df_X_test['datetime_1'].value_counts().sort_index()
df_X_test['datetime_2'].value_counts().sort_index()

# 2017-04-14    1170
df_X_d = df_X[df_X['datetime_0'] == '2017-04-14'].copy()
df_X_test_d = df_X_test[df_X_test['datetime_0'] == '2017-04-14'].copy()

# https://habr.com/company/nixsolutions/blog/425253/

def noise_columns(df, val):
    u = df.shape[0]
    return [col_name for col_name in df.columns if df[col_name].unique().shape[0] / u >= val]

noise_columns(df_X, 0.9)
noise_columns = ['number_{}'.format(i) for i in range(38, 87 + 1)] + ['number_36', 'number_37', 'number_11']
df_X_d.drop(noise_columns, axis=1, inplace=True)
df_X_test_d.drop(noise_columns, axis=1, inplace=True)

number_columns = [col_name for col_name in df_X_d.columns if col_name.startswith('number')]

datetime_columns = [col_name for col_name in df_X.columns if col_name.startswith('datetime')]
for cn in datetime_columns:
    df_X[cn] = pd.to_datetime(df_X[cn])
import itertools
for c1, c2 in list(itertools.combinations(datetime_columns, 2)):
    df_X['number_{}_{}'.format(c1, c2)] = (df_X[c1] - df_X[c2]).dt.days


df_X['datetime_0'] = pd.to_datetime(df_X['datetime_0'])
df_X['datetime_1'] = pd.to_datetime(df_X['datetime_1'])
df_X['datetime_2'] = pd.to_datetime(df_X['datetime_2'])

df_X['number_dt_01'] = (df_X['datetime_0'] - df_X['datetime_1']).dt.days
df_X['number_dt_20'] = (df_X['datetime_2'] - df_X['datetime_0']).dt.days
df_X['number_dt_21'] = (df_X['datetime_2'] - df_X['datetime_1']).dt.days

df_X_test['datetime_0'] = pd.to_datetime(df_X_test['datetime_0'])
df_X_test['datetime_1'] = pd.to_datetime(df_X_test['datetime_1'])
df_X_test['datetime_2'] = pd.to_datetime(df_X_test['datetime_2'])

df_X_test['number_dt_01'] = (df_X_test['datetime_0'] - df_X_test['datetime_1']).dt.days
df_X_test['number_dt_20'] = (df_X_test['datetime_2'] - df_X_test['datetime_0']).dt.days
df_X_test['number_dt_21'] = (df_X_test['datetime_2'] - df_X_test['datetime_1']).dt.days

used_columns = [col_name for col_name in df_X.columns if col_name.startswith('number')]

## Посмотрим на данные
used_columns = number_columns[:10]
df_X[used_columns].plot(marker='o')
df_X['target'].plot(marker='x', color='red')

# features from datetime
df_X = transform_datetime_features(df_X)
df_test = transform_datetime_features(df_test)

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
    for col_name in df_X_d.columns
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X[used_columns].values
y_value = df_X['target'].values

X_test = df_X_test[used_columns].values
y_test = df_X_test['target'].values
print('X_values shape {}'.format(X_values.shape))  # X_values shape (365, 39)
print('X_test shape {}'.format(X_test.shape))  # X_test shape (172, 39)

model = LGBMClassifier(n_estimators=70)
model.fit(X_values, y_value)
prediction = model.predict_proba(X_test)[:, 1]

result = df_X_test_d[['target']].copy()
result['prediction'] = prediction

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.04}'.format(metric))

# Проверка на обучении
result = df_X[['target']].copy()
result['prediction'] = model.predict_proba(X_values)[:, 1]

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.04}'.format(metric))

# Отправлено 0.9950
# 7708

y_true['prediction'].hist(bins=100)

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X[used_columns], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
