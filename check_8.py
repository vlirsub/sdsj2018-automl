# -*- coding: utf-8 -*-
# Анализ 8 набора данных

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
ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024

fd = os.path.join(DATA_DIR, 'check_8_c')
# Данные для обучени
train_csv = os.path.join(fd, r'train.csv')
# Данные для тестирования
test_csv = os.path.join(fd, r'test.csv')
# Правильные ответы
target_csv = os.path.join(fd, r'test-target.csv')

df = pd.read_csv(train_csv)
print('Train dataset read, shape {}'.format(df.shape))  # (143525, 878)
print('Train dataset memory usage {} MB'.format(df.memory_usage().sum() / 1024 / 1024))  # 961.4178466796875
df_y = df.target
df_X = df.drop('target', axis=1)
is_big = df_X.memory_usage().sum() > BIG_DATASET_SIZE
del df

# Данные для провекри
df_X_test = pd.read_csv(test_csv)
print('Test dataset read, shape {}'.format(df_X_test.shape))  # (61512, 877)
print('Test dataset memory usage {:} MB'.format(df_X_test.memory_usage().sum() / 1024 / 1024))  # 411.57557678222656 MB
# Данные с правильными ответами
df_y_test = pd.read_csv(target_csv)

df_X_test = pd.merge(df_X_test, df_y_test, on='line_id')
df_y_test = df_X_test.target
df_X_test = df_X_test.drop('target', axis=1)

print('Train dataset, shape {}'.format(df_X.shape))  # (143525, 877)
print('Test dataset, shape {}'.format(df_X_test.shape))  # (61512, 877)

print('Train true dataset, shape {}'.format(df_y.shape))  # (143525,)
print('Test true dataset, shape {}'.format(df_y_test.shape))  # (61512,)

# Преобразование колонок с датой
import datetime


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


df = df_X  # Для проверки

# Замена колонок string с датой на дату
datestr_columns = [col_name for col_name in df.columns if col_name.startswith('string')]
pd.to_datetime(df['string_1']).unique()


# Проверка содержется ли в колонке дата
def check_date() -> bool:


for d in df['string_20'].dropna().unique():
    l = len('2010-01-01')
    if len(d) != l:
        raise Exception('Dont date')

for col_name in datestr_columns:
    try:
        df[col_name].apply(lambda x: parse_dt(x))
        print('Дата ' + col_name)
    except:
        print('Не дата ' + col_name)
        pass


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        # df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        # df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        # df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
    return df


# drop constant features
constant_columns = [col_name for col_name in df_X.columns if df_X[col_name].nunique() == 1]
print('constant_columns: {}'.format(constant_columns))
df_X.drop(constant_columns, axis=1, inplace=True)
df_X_test.drop(constant_columns, axis=1, inplace=True)

# dict with data necessary to make predictions
model_config = {}
model_config['categorical_values'] = {}
model_config['is_big'] = True

# missing values
if any(df_X.isnull()):
    model_config['missing'] = True
    df_X.fillna(-1, inplace=True)
    df_X_test.fillna(-1, inplace=True)

new_feature_count = min(df_X.shape[1],
                        int(df_X.shape[1] / (df_X.memory_usage().sum() / BIG_DATASET_SIZE)))
# take only high correlated features
correlations = np.abs([
    np.corrcoef(df_y, df_X[col_name])[0, 1]
    for col_name in df_X.columns if col_name.startswith('number')
])
new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
df_X = df_X[new_columns].copy()
df_X_test = df_X_test[new_columns].copy()

#
number_columns = [col_name for col_name in df_X.columns if col_name.startswith('number')]
model_config['number_columns'] = number_columns
print('number_columns: {}'.format(number_columns))

#
id_columns = [col_name for col_name in df_X.columns if col_name.startswith('id')]
model_config['id_columns'] = id_columns
print('id_columns: {}'.format(id_columns))
#
datetime_columns = [col_name for col_name in df_X.columns if col_name.startswith('datetime')]
model_config['datetime_columns'] = datetime_columns
print('datetime_columns: {}'.format(datetime_columns))


# Колонки с шумом
def f_noise_columns(df, val):
    u = df.shape[0]
    return [col_name for col_name in df.columns if df[col_name].unique().shape[0] / u >= val]


noise_columns = f_noise_columns(df_X[number_columns], 0.95)
model_config['noise_columns'] = noise_columns
print('noise_columns: {}'.format(noise_columns))
df_X.drop(noise_columns, axis=1, inplace=True)
df_X_test.drop(noise_columns, axis=1, inplace=True)

# use only numeric columns
used_columns = [col_name for col_name in df_X.columns if
                col_name.startswith('number') or col_name.startswith('onehot')]
model_config['used_columns'] = used_columns
print('used_columns: {}'.format(used_columns))

# Данные для обучения
X_train = df_X[used_columns].values
y_train = df_y.values
X_test = df_X_test[used_columns].values
y_test = df_y_test.values
print('X_values shape {}'.format(X_train.shape))  # X_values shape (143525, 371)
print('X_test shape {}'.format(X_test.shape))  # X_test shape (61512, 371)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    # 'objective': 'regression' if args.mode == MODE_REGRESSION else 'binary',
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

model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 600)

prediction = model.predict(X_test)

result = pd.DataFrame()
result['target'] = y_test
result['prediction'] = prediction

metric = roc_auc_score(result['target'], result['prediction'])
print('roc auc: {:.4}'.format(metric))
# Отправлено 0.8835
# 0.8453 0.8461
# 0.8317

# df_X = transform_datetime_features(df_X)

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

# Есть даты и признаки
for sc in string_columns:
    print('{} {}'.format(sc, df_X[sc].unique()))
# string_0 ['МВС' 'Социальный' 'Массовый' nan 'VIP' 'Молодежный']
# string_1 ['3.1.6' '3.1.1' '3.1.11' '3.1.15' nan '3.1.26' '3.1.18' '3.1.12' '3.1.27' '3.1.8' '3.1.10' '3.1.20' '3.1.16' '3.1.5' '3.1.9' '3.1.23' '3.1.24'
#  '3.1.17' '3.1.19' '3.1.25' '3.1.2' '3.1.21' '3.1.7' '3.1.22']
# string_2 ['1.6_M' '1.1_M' '2.5_M' '2.10_F' nan '2.15_F' '2.15_M' '2.13_M' '2.6_F'
#  '2.10_M' '2.16_M' '2.1_F' '1.6_F' '2.3.C_F' '2.1_M' '2.16_F' '3.1.6_M'
#  '2.5_F' '2.11_F' '1.5_F' '1.1_F' '2.11_M' '2.2_F' '3.4.2_F' '1.8_F'
#  '2.2_M' '1.8_M' '2.6_M' '3.4.5_F' '1.5_M' '2.3.2_F' '2.12_F' '2.13_F'
#  '2.14_F' '2.3.2_M' '1.9_F' '1.2_M' '3.1.2_F' '2.3.C_M' '3.1.5_M'
#  '3.2.6_F' '3.4.1_F' '3.4.6_F' '3.2.3_M99' '1.2_F' '2.3.O_F' '3.2.5_F'
#  '3.2.2_F' '3.2.6_M' '3.4.2_M' '3.2.4_F' '3.4.4_M' '1.7_M' '3.2.2_M99'
#  '3.4.4_F' '3.1.6_F' '3.1.4_M' '1.7_F' '3.3.6_F' '2.12_M' '3.3.4_F'
#  '3.4.3_F' '3.3.6_M' '2.14_M' '3.2.3_F' '3.1.3_F' '1.9_M' '3.2.3_M'
#  '3.1.5_F' '3.4.6_M' '3.4.3_M' '3.1.4_F' '3.1.5_M99' '3.2.4_M' '3.1.1_F99'
#  '3.1.4_M99' '3.1.1_F' '3.2.1_F' '3.1.3_M' '2.3.O_M' '3.2.5_M' '3.2.2_M'
#  '3.2.6_M99' '3.1.1_M' '3.1.5_F99' '3.2.1_M' '3.1.2_M' '3.1.2_F99'
#  '3.3.4_M' '3.3.5_F' '3.2.1_M99' '3.1.4_F99' '3.2.5_M99' '3.4.2_F99'
#  '3.3.3_M' '3.1.6_M99' '3.3.2_M' '3.4.5_F99' '3.4.1_M99' '3.4.3_F99'
#  '3.4.6_M99' '3.1.3_F99' '3.4.1_F99' '3.3.2_F' '3.3.3_F' '3.4.6_F99'
#  '3.4.5_M' '3.4.1_M' '3.4.2_M99' '3.1.6_F99' '3.1.1_M99' '3.3.1_F'
#  '3.2.4_M99' '3.4.4_F99' '3.1.3_M99' '3.3.6_F99' '3.3.1_M' '3.1.2_M99'
#  '3.4.5_M99']
# string_3 ['M' 'F']
# string_4 ['NORM' 'PENS' 'YOUTH']
# string_5 [nan 'CITY_MLNR' 'CITY_OTHER' 'VILLAGE']
# string_6 ['CITY_OTHER' 'CITY_MLNR' 'VILLAGE' 'UNKNOWN']
# string_8 [nan '9042-PILIPENKO-YUV@TB42' 4161914.0 'ALADASHVILI-DD'
#  '8613-KRASNOV-KL@TB42.VVB.SBRF.RU' 'PONOMAREVA2-AY'
#  '6991_KLEYMENOVAES@PVB.SBRF.RU' '6991_SERGEYEVDA@PVB.SBRF.RU'
#  'SHIRMANDV@TB.FESB.RU' 'ISMAILOV-AS' 'YAKUSHINAAG_2578@SRB.SBRF.RU'
#  'MOROZOV-VS' 'KURGANOVA-EG' 'PAVLYUK-SA'
#  'MILITANMS_8608@KALUGA.SRB.LOCAL' 'SIZOV-MV' 'PICHUGINA-AA' 'YAKUSIK-SI'
#  'NOVOZHILOVA-VN' 'L8593_POLIKANIN_D@BEL.CCH.SBRF.R'
#  '5940SOLODNIKOVA-MM@ZSB.SBRF.RU' 'LISITSKAYAEV@BEL.CCH.SBRF.RU'
#  '8613-IVANOVA-IP@TB42.VVB.SBRF.RU' 'K8596_GADGIBEKOV_UG@BEL.CCH.SBRF'
#  'VASUTKINA-NP@TB31.VSB.SBRF.RU' 'MINCHENKOAI_8605@SRB.SBRF.RU'
#  '05-SHCHERBININA-DV@TERBANK.SIB.S' 'LAPSHIN-SS' 'FADEYEVA-OV'
#  'LYASHENKO-KN' '6991_SMIRNOVAYUA@PVB.SBRF.RU' 'PETUKHOV1-AD'
#  'BATYRBEKOVA-GM' '8613-GORDEEVA-EA@TB42' 'EMPLOYEE_ADMIN' 'SHABANOV1-DS'
#  'ZHUKKV@SVB.SBRF.RU']
# string_9 ['MVS' nan 'VIP']
# string_10 ['MVS' 'SOCIAL' 'MASS' 'VIP' 'YOUTH']
# string_11 ['MVS' 'SOCIAL' 'MASS' 'VIP' 'YOUTH' nan]
# string_12 ['TD' 'CC' 'DC' 'CA' 'MG' 'PL' nan 'SC' 'CR' 'MA']
# string_13 ['CA' 'DC' 'TD' nan 'PL' 'CR' 'MG' 'SC' 'MA' 'CC']
# string_14 ['IDEAL' 'CARD' 'DEPOSIT' 'NONE' 'CREDIT' 'CREDIT_DEPOSIT']
# string_15 ['IDEAL' 'CARD' 'DEPOSIT' 'NONE' 'CREDIT' 'CREDIT_DEPOSIT' nan]
# string_16 ['ACTIVE' 'INACTIVE_12+' 'INACTIVE_7_12' 'GONE' 'INACTIVE_3_6']
# string_17 ['ACTIVE' 'INACTIVE_7_12' 'INACTIVE_12+' 'GONE' 'INACTIVE_3_6' nan]
# string_18 ['000000000000' '111111111111' '000001111111' ... '011011111010' '101001011000' '101011010111']
# string_19 ['111111111111' '000000000000' '110111111111' '000000000001'
#  'XXXXXX000000' '111111111000' '111110000000' '110111010110'
#  'XXXXXXXXX000' nan '111111101110' '000001111111' '110111101111'
#  '010111011111' '110111011111' '110000000000' '000111111111'
#  '000000000011' '001111111111' 'X00000000000' '000000111111'
#  '111001111111' 'XXXXXXXXXXX0' '010111011110' '000000001111'
#  'XXXXXXX00000' '000000000111' '000111011111' '000011111111'
#  '111111111101' '111111110000' '111110011111' 'XXXXXX111111'
#  '000000011110' '110111011101' 'XXXX00000000' '111000000000'
#  '111111111100' 'XXX000000000' 'XXX111111110' 'XX0000000010'
#  '100000000000' '011111111111' '000001010110' '000000011111'
#  'XXXXXXXX0000' 'XXXXXX010110' '100011111111' '000000000100'
#  '111111010110' 'XXXXXXXXXX00' '010111111111' '111111111110'
#  '111011100000' '000010000100' 'XXXXXXX00111' 'XXXXXXX10110'
#  '001111110000' '000001011111' '000000000010' '111011111101'
#  'XXXXXXX11111' '001000000000' 'XX0000000000' '110011111111'
#  '000000011100' '110000000100' 'XXXXXXXX0111' '110111100111'
#  '000000100000' '110110000000' 'XXXXXX011111' '100000000010'
#  'XXXXXX000001' '000000001000' '000000111100' '000000010000'
#  '000000001011' '111111110111' '111100111111' '111111000000'
#  '111111101111' 'XXXXX0000000' 'XXXXXXX01111' 'XXX111111111'
#  '001000000100' '110000100000' 'XXXXXX001111' '000011111011'
#  'XXXXXXXX1111' '111100000000' '111111011101' '111111011111'
#  '000101111111' '110111110111' '110111110000' '110100000000'
#  '110111001111' 'XXXXXXXXX011' 'XXXXXX011101' '000100000000'
#  '110111010000' 'XXXXXXXXXXX1' '111111111001' '010000000001'
#  'X01111111111' '110001010110' '110110111111' 'XXXXXXX00010'
#  '110111111011' '110111111000' 'XXXXXXXXX001' '000100000100'
#  '110001000000' '000111010110' '000001000000' '110111000000'
#  '010000000011' '110111101100' 'XXX000010110' '000000000110'
#  'XXX001111111' 'XXXXX1000000' '001000111111' '110111111110'
#  '000011111100' '101111111111' '111111111011' '000000111110'
#  '000111101111' '000011111110' 'X00111111111' '110111111101'
#  '110111111100' '000001111110' 'XX0000111111' '010000000000'
#  '000000010110' 'X11111111111' 'XXXXXX000011' 'XXX000000100'
#  '000000010100' '000010000000' '100001000000' '110111101110'
#  '000000011101' 'XX0011111111' 'XX1111111111' '110111010111'
#  '110000000111' 'XXXXXXXX0011' '000000011000' '111111100000'
#  '000011000000' '010111010110' '000011010110' '110101011111'
#  'XXXXXXXXX110' 'XXXXXXXXXX10' 'XXX001011111' 'XXXXXX101111'
#  'XX0010000000' '110111010010' '000011011111' '110111011110'
#  '010111101111' '001111111100' 'XXXX00011111' '010010010010'
#  '000000101111' '000100111111' 'XXXXXX001110' '000011101111'
#  '000100000010' '110100000010' '110001111111' '111110111111'
#  'XX0111011110' '110000000001' 'XXXXXXXXX010' 'XXXXX0000011'
#  'XX0111111111' 'XXXXX0111111' '000000110000' '110111011100'
#  '110111100000' '000001000001' '010000000010' '111101111111'
#  '110111011001' 'XXXXXXXXX111' '000001110000' '000010000001'
#  '110111111001' '000001000010' '100000000100' 'XXX000000011'
#  '000010111101' 'XX0100000000' '010011011110' '110111110011'
#  '000000000101' '010111011101' 'XXXXX0000001' 'XXXX00000010'
#  'X00000000001' 'XXXXXXXXXX01' 'XXXXX0001110' 'XXX000011111'
#  '000100000001' '100000000011' '110000011111' '111111000010'
#  '100000111111' 'XXXXXX111110' 'XXXX01111111' '110111101000'
#  '110000000010' 'XXX000001111' 'XXXXXXXXXX11' '111011111111'
#  '011110000000' '111111110110' '000000110110' 'XXXXXX111001'
#  'XXXXXXX00001' '000001111000' '111111100111' 'XXXXXXX00011'
#  '010111110000' '010000000100' 'XXX111101111' '110101010110'
#  '010100111111' 'XXXXXX000111' '110000001111' 'XXXXX1111111'
#  '000001011110' '110100000100' '110111011000' 'XXX111011111'
#  'XX1111101111' 'X00000000111' '010001010110' '100000000111'
#  '110101111111' '100000000001' 'XXXX11111111' '000000111000'
#  'XXXXXXX01110' 'X00000000011' 'XXX000000001' '000011110101'
#  '000011011110' 'XXXXXXX01001' '000001101111' 'XXX000111111'
#  '111111001111' 'XX0000000001' 'XX0000001111' 'XXXXXX011110'
#  '000000001110' '111111110001' '100111111111' '111000000001'
#  'XXXXXXXX1000' '111011111100' '010001011111' '111011111000'
#  'XXXX00000111' 'XXXXXXXX0010' '100000001111' '000010000010'
#  '000000010010' 'XXXXX0001111' 'XXXXXXX11101' 'X00111011111'
#  '111110000111' '110111101001' '000111011110' '110000111111'
#  'XXXXXXXX0001' '110111010100' 'XXX100000000' 'X00011111111'
#  '000000100100' '010010000000' '110111110110' '000000001101'
#  '000001111001' 'XXX011111111' '011111111100' '000111000000'
#  '110000100001' 'XXXX11010110' '111011110101' '011111000000'
#  'XXXX00001111' '110100001111' '011111010110' '110111101011'
#  '110110001111' '001111000000' '110011011101' '000001010000'
#  'XXX111111000' '011111100000' 'XXXX00000001' '101100111111'
#  '110010011101' 'XXXXXX000100' '000001000100' '111110001111'
#  '110111010011' '111111000111' '110011011111' 'X00000111111'
#  '001111111110' '000111010111' '000100000011' '010111110111'
#  '110111010101' '001001111111' '001111011111' '000011111000'
#  'XXXXX0010110' 'XXXX00111111' '000001011101' 'XXX000000111'
#  '111011011111' 'XXXX00011110' 'XXX111010110' '000111100000'
#  '000010011101' '111000111101' 'XXXXXXX00100' '000100111100'
#  '001111111000' '000110000000' '000011111101' '010000001111'
#  '110111010001' '000111111110' '111011111110' '000111011101'
#  '010111000000' '010000000111' '111111110010' '110101110000'
#  '000000111001' 'XXXXXX111000' '001001000000' 'XX0111101111'
#  '010100000000' 'X00000001111' '001000000001' '000000010011'
#  '000001111100' '110110011111' '001111111001' 'XXXXX0011111'
#  'XXXXXX111100' '111100000011' '110011101111' '001000000010'
#  'XXXXXXXX1101' '100111111110' 'X00011111110' '100111111100'
#  '111111100011' '111111101101' 'XXXX10000000' '000000100010'
#  '000111111101' '100011101111' '011000000000' '110000000011'
#  'XXXXX0001100' '110101001111' '010111111000' '111110000010'
#  '000000001100' 'X00100000000' 'X00001111111' '110001011111'
#  'X00001010110' '111110001000' '010111111110' '111010000000'
#  '111010001110' '000010111111' '001110111111' 'XXXXXX001000'
#  'XXXXXXX11110' 'XX0000000011' 'XX0000000111' '111111010111'
#  '000111110000' 'XXXXXX000010' '000001110101' '011100000000'
#  '110011010110' '010011101000' '001011111111' '001100000100'
#  'XXX111011110' '011111111000' 'X10111111111' '000111101000'
#  'XXXXXXXX0100' '011011111101' 'XXXXX0000111' '001100000000'
#  '000110111111' '001010011111' '100000111110' 'XX0000000100'
#  '110111000011' '010111110011' '100011011111' '111011110100'
#  '111111110101' '110100111111' '110111110001' '111111110011'
#  '010111111101' '101111111101' 'XXXXXXX11100' 'XXXXXX000110'
#  '000010010010' '010000111001' 'XXXXXX110000' '010110000000'
#  '101111100000' 'XX0001111111' '000001111101' '000001000011'
#  '110100011111' 'XXXX01110000' '001111111011' '010101101111'
#  '100000011111' 'XXXXXXXX0110' '000000001001' '000001000110'
#  '111001111100' '010111011000' 'XXXXXX101101' 'XXXX01000000'
#  '010111011100' '010111111011' 'XXX000110110' '101011111101'
#  '111111100001' '110000101111' '110001000010' '000001100000'
#  '011111110000' 'XXXXXX111101' '010000011111' '011011111111'
#  'XXXXXX100000' '000010010110' '010011111111' 'XXX111010111'
#  '101011110110' '000111100011' '010001000000' '000011111001'
#  '111111010000' 'X00111011101' 'XXXXXX100001' '101000000000'
#  '110101101111' '001111111101' '110111110010' '100001111100'
#  '000010000110' '111011111001' '100111100000' '110000111000'
#  '001111100000' '000000010111' '100001011111' '010110100010'
#  '110000000110']
# string_20 [nan '2016-10-02' '2016-10-04' '2016-10-24' '2016-10-25' '2016-10-20'
#  '2016-10-22' '2016-10-30' '2016-10-13' '2016-10-10' '2016-10-01'
#  '2016-10-12' '2016-10-29' '2016-10-05' '2016-10-31' '2016-10-28'
#  '2016-10-03' '2016-10-23' '2016-10-18' '2016-10-14' '2016-10-07'
#  '2016-10-08' '2016-10-19' '2016-10-15' '2016-10-11' '2016-10-21'
#  '2016-10-06' '2016-10-27' '2016-10-17' '2016-10-16' '2016-10-26'
#  '2016-10-09']
# string_21 [nan '2016-10-02' '2016-10-04' '2016-10-24' '2016-10-25' '2016-10-20'
#  '2016-10-22' '2016-10-30' '2016-10-13' '2016-10-10' '2016-10-01'
#  '2016-10-12' '2016-10-29' '2016-10-21' '2016-10-05' '2016-10-31'
#  '2016-10-28' '2016-10-03' '2016-10-23' '2016-10-18' '2016-10-14'
#  '2016-10-07' '2016-10-19' '2016-10-26' '2016-10-15' '2016-10-27'
#  '2016-10-06' '2016-10-16' '2016-10-08' '2016-10-17' '2016-10-11'
#  '2016-10-09']
# string_22 [nan 'GOLD' 'PREMIER' 'VIP']
# string_23 [nan 'GOLD' 'PREMIER' 'VIP']
# string_24 [nan 'GOLD' 'PREMIER' 'VIP']
# string_25 ['ACTIVE' 'INACTIVE_12+' 'INACTIVE_7_12' 'GONE' 'INACTIVE_3_6']
# string_26 ['ACTIVE' 'INACTIVE_7_12' 'INACTIVE_12+' 'GONE' 'INACTIVE_3_6' '0']
# string_27 ['Автолюбители' 'Активно путешествующие' 'Кэшевики без покупок'
#  'Универсалы' nan 'Любители ресторанов/развлечений' 'Кэшевики с покупками'
#  'Покупатели Direct Marketing' 'Прекрасный пол' 'Любители HighTech']
# string_28 ['Северо-Западный банк' 'Московский банк' 'Среднерусский банк'
#  'Западно-Сибирский банк' 'Западно-Уральский банк' 'Уральский банк'
#  'Центрально-Черноземный банк' 'Волго-Вятский банк' 'Поволжский банк'
#  'Сибирский банк' 'Байкальский банк' 'Северный банк' 'Юго-Западный банк'
#  nan 'Дальневосточный банк']
# string_29 ['Северо-Западный банк' 'Московский банк' 'Среднерусский банк'
#  'Западно-Сибирский банк' 'Западно-Уральский банк' 'Волго-Вятский банк'
#  'Уральский банк' 'Центрально-Черноземный банк' 'Поволжский банк'
#  'Сибирский банк' 'Байкальский банк' 'Северный банк' 'Юго-Западный банк'
#  nan 'Дальневосточный банк']
# string_30 [nan 'Недоставленная корреспонденция_моб. телефон'
#  'Недоставленная корреспонденция_email'
#  'Недоставленная корреспонденция_почта' 'Блокирующий, не террорист']
# string_31 ['NORM' 'PENS' 'PREPENS']


for с in datetime_columns:
    print('{} {}'.format(с, df_X[с].unique()))

[]
for с in id_columns:
    print('{} {}'.format(с, df_X[с].unique()))

for с in number_columns:
    print('{} {}'.format(с, df_X[с].unique()))

df_X['number_0'].value_counts()
# 0.0    138545
#  1.0      4980

y_true['target'].value_counts()
# 0.0    59377
# 1.0     2135


df_X['number_0'].hist()
df_X['datetime_1'].hist()
df_X['id_0'].hist()
df_X['number_25'].hist(bins=100)

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
# Отправлено 0.8835
# 0.8453 0.8461
# 0.8317

result['prediction'].hist(bins=100)

# %% Важность признаков
fi = pd.DataFrame(list(zip(df_X[used_columns], model.feature_importances_)), columns=('clm', 'imp'))
fi.sort_values(by='imp', inplace=True, ascending=False)

# eof
