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
from sklearn.pipeline import make_pipeline

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
    # X_train = scaler.fit_transform(X_train)
    # model_config['scaler'] = scaler

    # fitting
    model_config['mode'] = args.mode
    if args.mode == MODE_regression:
        # Подбор модели
        Scores = list()
        model0 = make_pipeline(StandardScaler(), Ridge())
        for model in [model0, Ridge(), LGBMRegressor(n_estimators=70), LGBMRegressor(n_estimators=50),
                      LGBMRegressor(n_estimators=100)]:
            model.fit(X_train, y_train)
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            score = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=1, scoring='neg_mean_squared_error',
                                    verbose=0)

            Scores.append((abs(score.mean()), model))
        Scores.sort(key=lambda k: k[0])

        model = Scores[0][1]
    else:
        # model = LogisticRegression()
        # model = LGBMClassifier(n_estimators=70)
        # model.fit(X_train, y_train)
        # Подбор модели
        Scores = list()
        # model0 = make_pipeline(StandardScaler(), Ridge())
        for model in [LGBMClassifier(n_estimators=50), LGBMClassifier(n_estimators=70),
                      LGBMClassifier(n_estimators=100)]:
            model.fit(X_train, y_train)
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            score = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=1, scoring='roc_auc',
                                    verbose=0)

            Scores.append((abs(score.mean()), model))
        Scores.sort(key=lambda k: k[0])

        model = Scores[0][1]

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
