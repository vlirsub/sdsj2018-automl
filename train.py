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
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
ONEHOT_MAX_UNIQUE_VALUES = 32
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

    datetime_columns = [col_name
                        for col_name in df_X.columns
                        if col_name.startswith('datetime')]
    model_config['datetime_columns'] = datetime_columns
    print('datetime_columns {}'.format(datetime_columns))

    id_columns = [col_name
                  for col_name in df_X.columns
                  if col_name.startswith('id')]
    model_config['id_columns'] = id_columns
    print('id_columns {}'.format(id_columns))

    number_columns = [col_name
                      for col_name in df_X.columns
                      if col_name.startswith('number')]
    model_config['number_columns'] = number_columns

    if args.mode == MODE_regression \
            and len(datetime_columns) > 0 \
            and len(id_columns) > 0:

        number_columns = [
            col_name
            for col_name in df_X.columns
            if col_name.startswith('number')
        ]

        model_config['number_columns'] = number_columns

        print('Add shift columns for numeric')

        # def f_trans(x):
        #     for cn in number_columns:
        #         for i in range(1, 4 * 7 + 1):
        #             cn_shift = '{}_s{}'.format(cn, i)
        #             x[cn_shift] = x[cn].shift(i)
        #             x[cn_shift].fillna(-1, inplace=True)
        #     return x

        # def f_trans(x):
        #     x['number_23_s'] = x['number_23'].shift(-1).fillna(0)
        #
        #     return x
        #
        # df_X = df_X[id_columns + ['line_id'] + number_columns].groupby(id_columns).apply(f_trans)

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

    # fitting
    model_config['mode'] = args.mode
    if args.mode == MODE_regression:
        # Подбор модели
        Scores = list()
        for model in [Ridge(), LGBMRegressor(n_estimators=70)]:
            model.fit(X_train, y_train)
            kfold = KFold(n_splits=3, shuffle=True, random_state=0)
            score = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=1, scoring='neg_mean_squared_error',
                                    verbose=0)

            print('X {} y {} score: {} mean: {}'.format(X_train.shape, y_train.shape, score.round(2), score.mean()))
            Scores.append((abs(score.mean()), model))
        Scores.sort(key=lambda k: k[0])

        model = Scores[0][1]
        print(Scores)

    else:
        # model = LogisticRegression()
        model = LGBMClassifier(n_estimators=70)
        model.fit(X_train, y_train)

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
