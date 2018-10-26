import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024

MODE_REGRESSION = 'regression'


def main(args):
    start_time = time.time()

    df = pd.read_csv(args.train_csv)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    is_big = df_X.memory_usage().sum() > BIG_DATASET_SIZE

    print('Dataset read, shape {}'.format(df_X.shape))

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

        # categorical encoding
        categorical_values = {}
        for col_name in list(df_X.columns):
            col_unique_values = df_X[col_name].unique()
            if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
                categorical_values[col_name] = col_unique_values
                for unique_value in col_unique_values:
                    df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)
        model_config['categorical_values'] = categorical_values

        # missing values
        if any(df_X.isnull()):
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

    number_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number')
    ]
    model_config['number_columns'] = number_columns

    id_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('id')
    ]
    model_config['id_columns'] = id_columns
    print('id_columns: {}'.format(id_columns))

    datetime_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('datetime')
    ]
    model_config['datetime_columns'] = datetime_columns
    print('datetime_columns: {}'.format(datetime_columns))

    # Колонки с шумом
    # def f_noise_columns(df, val):
    #     u = df.shape[0]
    #     return [col_name for col_name in df.columns if df[col_name].unique().shape[0] / u >= val]
    #
    # noise_columns = f_noise_columns(df_X[number_columns], 0.9)
    # model_config['noise_columns'] = noise_columns
    # print('noise_columns: {}'.format(noise_columns))
    # df_X.drop(noise_columns, axis=1, inplace=True)

    if len(id_columns) > 0 and len(datetime_columns) > 0 and args.mode == MODE_REGRESSION:
        # # check_3
        def f_trans(x):
            for cn in number_columns:
                x['{}_s{}'.format(cn, -1)] = x[cn].shift(-1).fillna(0)

            return x

        df_X = df_X[id_columns + ['line_id'] + number_columns].groupby(id_columns).apply(f_trans)

    if 3 <= len(datetime_columns) <= 10:
        # check_4
        print('Add delta datetime columns')
        for cn in datetime_columns:
            df_X[cn] = pd.to_datetime(df_X[cn])
        import itertools
        for c1, c2 in list(itertools.combinations(datetime_columns, 2)):
            df_X['number_{}_{}'.format(c1, c2)] = (df_X[c1] - df_X[c2]).dt.days

    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]
    model_config['used_columns'] = used_columns

    X_train = df_X[used_columns].values
    y_train = df_y.values
    # scaling
    # scaler = StandardScaler(copy=False)
    # df_X = scaler.fit_transform(df_X)
    # model_config['scaler'] = scaler

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression' if args.mode == MODE_REGRESSION else 'binary',
        # 'objective': 'binary',
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
        'num_threads': 4,
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 600)

    # fitting
    model_config['mode'] = args.mode
    # if args.mode == MODE_REGRESSION:
    #     # Подбор модели
    #     # check_2
    #     Scores = list()
    #     for model in [Ridge(), LGBMRegressor(n_estimators=70)]:
    #         model.fit(X_train, y_train)
    #         kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    #         score = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=1, scoring='neg_mean_squared_error',
    #                                 verbose=0)
    #
    #         print('X {} y {} score: {} mean: {}'.format(X_train.shape, y_train.shape, score.round(2), score.mean()))
    #         Scores.append((abs(score.mean()), model))
    #     Scores.sort(key=lambda k: k[0])
    #
    #     model = Scores[0][1]
    #     print(Scores)
    #
    # else:
    #     # model = RidgeClassifier()
    #     # model = LGBMClassifier(n_estimators=70)
    #     # model.fit(X_train, y_train)
    #
    #     Scores = list()
    #     for model in [RidgeClassifier(), LGBMClassifier(n_estimators=70)]:
    #         model.fit(X_train, y_train)
    #         kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    #         score = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=1, scoring='roc_auc',
    #                                 verbose=0)
    #
    #         print('X {} y {} score: {} mean: {}'.format(X_train.shape, y_train.shape, score.round(2), score.mean()))
    #         Scores.append((abs(score.mean()), model))
    #     Scores.sort(key=lambda k: k[0], reverse=True)
    #
    #     model = Scores[0][1]
    #     print(Scores)

    model_config['model'] = model

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
