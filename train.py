import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.preprocessing import StandardScaler

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
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
    model_config['categorical_values'] = {}
    model_config['is_big'] = is_big
    print('is_big {}'.format(is_big))
    model_config['is_null_taget'] = None

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
        model_config['is_null_taget'] = df_y[df_y == 0].shape[0] > 0
        print('is_null_taget {}'.format(model_config['is_null_taget']))

        if len(datetime_columns) > 0 and len(id_columns) <= 0:
            model_config['is_work'] = True
        else:
            model_config['is_work'] = False
    else:
        model_config['is_work'] = False

    print('is_work {}'.format(model_config['is_work']))

    if model_config['is_work']:
        if is_big:
            # missing values
            if any(df_X.isnull()):
                model_config['missing'] = True
                df_X.fillna(-1, inplace=True)

            new_feature_count = min(df_X.shape[1],
                                    int(df_X.shape[1] / (df_X.memory_usage().sum() / BIG_DATASET_SIZE)))
            # take only high correlated features
            correlations = np.abs([
                np.corrcoef(df_y, df_X[col_name])[0, 1]
                for col_name in df_X.columns if col_name.startswith('number')
            ])
            new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
            df_X = df_X[new_columns]

        else:
            # features from datetime

            df_X = transform_datetime_features(df_X)

            #categorical encoding
            categorical_values = {}
            for col_name in list(df_X.columns):
                col_unique_values = df_X[col_name].unique()
                if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
                    categorical_values[col_name] = col_unique_values
                    for unique_value in col_unique_values:
                        df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)
            model_config['categorical_values'] = categorical_values
            print('categorical_values {}'.format(categorical_values))

            # missing values
            if any(df_X.isnull()):
                model_config['missing'] = True
                df_X.fillna(-1, inplace=True)

        # use only numeric columns
        used_columns = [
            col_name
            for col_name in df_X.columns
            if col_name.startswith('number') or col_name.startswith('onehot')
        ]
        model_config['used_columns'] = used_columns
        print('used_columns {}'.format(used_columns))

        # Данные для обучения
        X_values = df_X[used_columns].values

        # scaling
        #scaler = StandardScaler(copy=False)
        #df_X = scaler.fit_transform(df_X)
        #model_config['scaler'] = scaler

        # fitting
        model_config['mode'] = args.mode
        if args.mode == 'regression':
            model = Ridge()
            #model = LGBMRegressor(n_estimators=70)
            # Данные для классификации
            df_y_c = df_y.copy()
            df_y_c = (df_y_c > 0).astype(np.int8)
            model_c = LogisticRegression()
            model_c.fit(X_values, df_y_c)
            model_config['model_c'] = model_c
        else:
            #model = LogisticRegression()
            model = LGBMClassifier(n_estimators=70)

        model.fit(X_values, df_y)
        model_config['model'] = model

        # RMSE: 113.1 86.21

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



