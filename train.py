import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.preprocessing import StandardScaler

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024

MODE_classification = r'classification'
MODE_regression = r'regression'


def main(args):
    start_time = time.time()

    df = pd.read_csv(args.train_csv)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    is_big = df_X.memory_usage().sum() > BIG_DATASET_SIZE

    print('Dataset read, shape {}'.format(df_X.shape))
    print('Dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))

    # drop constant features
    constant_columns = [
        col_name
        for col_name in df_X.columns
        if df_X[col_name].nunique() == 1
    ]
    df_X.drop(constant_columns, axis=1, inplace=True)

    # dict with data necessary to make predictions
    model_config = {}
    model_config['mode'] = args.mode
    model_config['categorical_values'] = {}
    model_config['is_big'] = is_big
    print('is_big {}'.format(is_big))
    model_config['is_null_target'] = None

    if args.mode == MODE_regression:
        # Есть ли колонки с датой
        datetime_columns = [
            col_name
            for col_name in df.columns
            if col_name.startswith('datetime')
        ]
        # Есть ли колонки с id
        id_columns = [
            col_name
            for col_name in df.columns
            if col_name.startswith('id')
        ]
        # Есть ли 0 в target
        model_config['is_null_target'] = df_y[df_y == 0].shape[0] > 0
        if len(datetime_columns) > 0 and len(id_columns) <= 0:
            model_config['is_work'] = True
        else:
            model_config['is_work'] = False
        print('is_null_target {}'.format(model_config['is_null_target']))
    else:
        model_config['is_work'] = False

    print('is_work {}'.format(model_config['is_work']))

    if model_config['is_work']:
        if model_config['is_null_target']:
            # Для классификации составляем датасет из признаков есть target или нет
            df_X_c = df.copy()
            # df_X_c.shape  # 365, 42
            df_X_c['target'] = (df_X_c['target'] > 0).astype(np.int8)

            df_y_c = df_X_c[['target']].copy()
            # df_y_c.shape  # 365, 1
            df_X_c = df_X_c.drop('target', axis=1)
            # df_X_c.shape  # 365, 41

            # Полный набор данных
            df_X = df.copy()
            df_y = df_X[['target']].copy()
            df_X = df_X.drop('target', axis=1)
            # df_X.shape  # 365, 41
            # df_y.shape  # 365, 1
            # В начале классифицируем по признаку надо ли делать регресию или нет, target > 0
            used_columns = [
                col_name
                for col_name in df_X_c.columns
                if col_name.startswith('number') or col_name.startswith('onehot')
            ]
            X_values = df_X_c[used_columns].values

            model_c = LogisticRegression()
            model_c.fit(X_values, df_y_c['target'])
            model_config['model_c'] = model_c

            # Обучение на полном наборе данных для регрессии
            X_values = df_X[used_columns].values

            model_r = Ridge()
            model_r.fit(X_values, df_y['target'].interpolate().bfill())

            model_config['used_columns'] = used_columns
            model_config['model_r'] = model_r

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    main(args)
