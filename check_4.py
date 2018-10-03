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
print('Train dataset memory usage {} MB'.format(df.memory_usage().sum() / 1024 / 1024))

# Данные для провекри
df_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df.shape))
print('Test dataset memory usage {:.3} MB'.format(df_test.memory_usage().sum() / 1024 / 1024))
# Данные с правильными ответами
y_true = pd.read_csv(target_csv)

# Для классификации составляем датасет из признаков есть target или нет
df_X = df.copy()
df_X.shape  # 114130, 143

df_y = df_X[['target']].copy()
df_y.shape  # 114130, 1
df_X = df_X.drop('target', axis=1)
df_X.shape  # 114130, 142
# Правильные ответы
y_test = y_true.copy()
y_test.shape  # 45385, 2

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
    if col_name.startswith('number') or col_name.startswith('onehot')
]
X_values = df_X[used_columns].values
X_test = df_test[used_columns].values
print('X_values shape {}'.format(X_values.shape))  # X_values shape (365, 39)
print('X_test shape {}'.format(X_test.shape))  # X_test shape (172, 39)
#used_columns ['number_0', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'number_7', 'number_8', 'number_9', 'number_10', 'number_11', 'number_12', 'number_13', 'number_14',
# 'number_15', 'number_16', 'number_17', 'number_18', 'number_19', 'number_20', 'number_21', 'number_22', 'number_23', 'number_24', 'number_25', 'number_26', 'number_27', 'number_28', 'number_29',
# 'number_30', 'number_31', 'number_32', 'number_33', 'number_34', 'number_35', 'number_36', 'number_37', 'number_38', 'number_39', 'number_40', 'number_41', 'number_42', 'number_43', 'number_44',
# 'number_45', 'number_46', 'number_47', 'number_48', 'number_49', 'number_50', 'number_51', 'number_52', 'number_53', 'number_54', 'number_55', 'number_56', 'number_57', 'number_58', 'number_59',
# 'number_60', 'number_61', 'number_62', 'number_63', 'number_64', 'number_65', 'number_66', 'number_67', 'number_68', 'number_69',
# 'number_70', 'number_71', 'number_72', 'number_73', 'number_74', 'number_75', 'number_76', 'number_77', 'number_78', 'number_79',
# 'number_80', 'number_81', 'number_82', 'number_83', 'number_84', 'number_85', 'number_86', 'number_87',
# 'number_weekday_datetime_0', 'number_month_datetime_0', 'number_day_datetime_0', 'number_hour_datetime_0', 'number_hour_of_week_datetime_0', 'number_minute_of_day_datetime_0',
# 'number_weekday_datetime_1', 'number_month_datetime_1', 'number_day_datetime_1', 'number_hour_datetime_1', 'number_hour_of_week_datetime_1', 'number_minute_of_day_datetime_1',
# 'number_weekday_datetime_2', 'number_month_datetime_2', 'number_day_datetime_2', 'number_hour_datetime_2', 'number_hour_of_week_datetime_2', 'number_minute_of_day_datetime_2',
# 'onehot_number_7=1.0', 'onehot_number_7=0.0', 'onehot_number_7=3.0', 'onehot_number_7=2.0', 'onehot_number_7=5.0', 'onehot_number_7=4.0',
# 'onehot_number_10=7.0', 'onehot_number_10=nan', 'onehot_number_10=11.0', 'onehot_number_10=1.0', 'onehot_number_10=10.0', 'onehot_number_10=5.0', 'onehot_number_10=9.0', 'onehot_number_10=12.0', 'onehot_number_10=6.0', 'onehot_number_10=14.0', 'onehot_number_10=8.0', 'onehot_number_10=2.0', 'onehot_number_10=15.0', 'onehot_number_10=3.0', 'onehot_number_10=4.0', 'onehot_number_10=13.0', 'onehot_number_10=0.0',
# 'onehot_number_12=3.0', 'onehot_number_12=1.0', 'onehot_number_12=8.0', 'onehot_number_12=5.0', 'onehot_number_12=2.0', 'onehot_number_12=6.0', 'onehot_number_12=4.0', 'onehot_number_12=7.0', 'onehot_number_12=0.0', 'onehot_number_12=9.0', 'onehot_number_12=10.0', 'onehot_number_12=14.0', 'onehot_number_12=11.0', 'onehot_number_12=12.0', 'onehot_number_12=13.0',
# 'onehot_number_18=-1.0871061812963851', 'onehot_number_18=0.6006391648700256', 'onehot_number_18=0.2317188192314223', 'onehot_number_18=0.06808583530226567', 'onehot_number_19=-0.00672539533019319', 'onehot_number_19=-0.07728638818211044', 'onehot_number_19=0.2758232833541581', 'onehot_number_19=-2.1833356297906747', 'onehot_number_19=-0.7515479900119882', 'onehot_number_19=-0.0839007318679354', 'onehot_number_20=0.3548350013149225', 'onehot_number_20=0.07018409330034325', 'onehot_number_20=-0.09180522598923643', 'onehot_number_20=-0.4422177920208029', 'onehot_number_21=0.004429797285981609', 'onehot_number_21=0.2561043602086837', 'onehot_number_21=-0.27747644092262724', 'onehot_number_21=0.4791880708150199', 'onehot_number_21=-0.7235904568185785', 'onehot_number_21=7.583629130871626', 'onehot_number_22=-0.2821317280514669', 'onehot_number_22=0.3648236092926316', 'onehot_number_22=-0.004337721741874449', 'onehot_number_22=-1.0927850173573308', 'onehot_number_23=0.2288323103681056', 'onehot_number_23=-0.9955697515059281', 'onehot_number_23=-0.3295544089237658', 'onehot_number_23=-0.656386826999885', 'onehot_number_24=-0.2596385049652115', 'onehot_number_24=0.2384490318832964', 'onehot_number_24=-0.09600883127199793', 'onehot_number_24=0.11891990974531985', 'onehot_number_25=0.05674906986545709', 'onehot_number_25=0.5716107129433511', 'onehot_number_25=-0.38234543536848986', 'onehot_number_25=-0.1968369296797297', 'onehot_number_26=-0.6912395269337647', 'onehot_number_26=0.19129965408893607', 'onehot_number_26=0.2653845385356593', 'onehot_number_26=-2.050443392506804', 'onehot_number_27=-0.0001279792955197721', 'onehot_number_27=-0.35206326322789105', 'onehot_number_27=0.2778450700243265', 'onehot_number_27=-0.8218674617628474', 'onehot_number_27=-0.2523152138837401', 'onehot_number_27=-0.5138974698405093', 'onehot_number_28=0.24395520538206725', 'onehot_number_28=-0.02602985357192339', 'onehot_number_28=-0.2947629257694702', 'onehot_number_28=0.571014281400421', 'onehot_number_28=-0.1272391660451408', 'onehot_number_28=-0.6045401811043346', 'onehot_number_29=0.11673306162885624', 'onehot_number_29=0.0751830130423388', 'onehot_number_29=-0.03052134692383932', 'onehot_number_29=-0.2470647152889097', 'onehot_number_30=-0.544914400526879', 'onehot_number_30=0.5966546411282899', 'onehot_number_30=-1.3034847311622442', 'onehot_number_30=0.1525158657680117', 'onehot_number_31=-0.7103772251320365', 'onehot_number_31=0.9409718295255244', 'onehot_number_31=-0.575164909822376', 'onehot_number_31=0.4753025662013369', 'onehot_number_31=0.11574013336310965', 'onehot_number_31=-0.18415746834264687', 'onehot_number_31=-0.7748410149804514', 'onehot_number_32=-0.37820302476531104', 'onehot_number_32=-0.09200905247076607', 'onehot_number_32=0.4972710710961178', 'onehot_number_32=0.27528416423741897', 'onehot_number_33=-0.3259817472059873', 'onehot_number_33=0.15104220414442965', 'onehot_number_33=-0.5070726240803399', 'onehot_number_33=0.10921689417364888', 'onehot_number_33=-0.3515079231381187', 'onehot_number_33=-0.9321287615150696', 'onehot_number_34=0.08287292001461476', 'onehot_number_34=-0.4141599680067063', 'onehot_number_34=-0.3475446695398697', 'onehot_number_34=-0.581523190236276', 'onehot_number_35=0.005494282302663543', 'onehot_number_35=0.4502413923512823', 'onehot_number_35=-0.19923515532456665', 'onehot_number_35=-0.0635751801904112', 'onehot_number_weekday_datetime_0=4', 'onehot_number_weekday_datetime_0=3', 'onehot_number_weekday_datetime_0=5', 'onehot_number_weekday_datetime_0=0', 'onehot_number_weekday_datetime_0=2', 'onehot_number_weekday_datetime_0=6', 'onehot_number_weekday_datetime_0=1', 'onehot_number_month_datetime_0=3', 'onehot_number_month_datetime_0=4', 'onehot_number_month_datetime_0=9', 'onehot_number_month_datetime_0=5', 'onehot_number_month_datetime_0=7', 'onehot_number_month_datetime_0=8', 'onehot_number_month_datetime_0=6', 'onehot_number_month_datetime_0=2', 'onehot_number_hour_of_week_datetime_0=96', 'onehot_number_hour_of_week_datetime_0=72', 'onehot_number_hour_of_week_datetime_0=120', 'onehot_number_hour_of_week_datetime_0=0', 'onehot_number_hour_of_week_datetime_0=48', 'onehot_number_hour_of_week_datetime_0=144', 'onehot_number_hour_of_week_datetime_0=24', 'onehot_number_weekday_datetime_1=0', 'onehot_number_weekday_datetime_1=4', 'onehot_number_weekday_datetime_1=3', 'onehot_number_weekday_datetime_1=1', 'onehot_number_weekday_datetime_1=2', 'onehot_number_weekday_datetime_1=6', 'onehot_number_weekday_datetime_1=5', 'onehot_number_month_datetime_1=3', 'onehot_number_month_datetime_1=8', 'onehot_number_month_datetime_1=4', 'onehot_number_month_datetime_1=6', 'onehot_number_month_datetime_1=5', 'onehot_number_month_datetime_1=7', 'onehot_number_month_datetime_1=2', 'onehot_number_month_datetime_1=9', 'onehot_number_hour_of_week_datetime_1=0', 'onehot_number_hour_of_week_datetime_1=96', 'onehot_number_hour_of_week_datetime_1=72', 'onehot_number_hour_of_week_datetime_1=24', 'onehot_number_hour_of_week_datetime_1=48', 'onehot_number_hour_of_week_datetime_1=144', 'onehot_number_hour_of_week_datetime_1=120', 'onehot_number_weekday_datetime_2=1', 'onehot_number_weekday_datetime_2=3', 'onehot_number_weekday_datetime_2=4', 'onehot_number_weekday_datetime_2=0', 'onehot_number_weekday_datetime_2=6', 'onehot_number_weekday_datetime_2=2', 'onehot_number_weekday_datetime_2=5', 'onehot_number_month_datetime_2=4', 'onehot_number_month_datetime_2=10', 'onehot_number_month_datetime_2=7', 'onehot_number_month_datetime_2=6', 'onehot_number_month_datetime_2=5', 'onehot_number_month_datetime_2=3', 'onehot_number_month_datetime_2=9', 'onehot_number_month_datetime_2=8', 'onehot_number_month_datetime_2=12', 'onehot_number_month_datetime_2=2', 'onehot_number_month_datetime_2=11', 'onehot_number_month_datetime_2=1', 'onehot_number_hour_of_week_datetime_2=24', 'onehot_number_hour_of_week_datetime_2=72', 'onehot_number_hour_of_week_datetime_2=96', 'onehot_number_hour_of_week_datetime_2=0', 'onehot_number_hour_of_week_datetime_2=144', 'onehot_number_hour_of_week_datetime_2=48', 'onehot_number_hour_of_week_datetime_2=120']

model = LGBMClassifier(n_estimators=100)
model.fit(X_values, df_y)
prediction = model.predict_proba(X_test)[:, 1]

result = y_true.copy()
result['prediction'] = prediction

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.4}'.format(metric))
# Отправлено 0.9950
#7708

y_true['prediction'].hist(bins=100)

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X[used_columns], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
