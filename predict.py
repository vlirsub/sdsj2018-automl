import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

MODE_classification = r'classification'
MODE_regression = r'regression'


def main(args):
    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Test Dataset read, shape {}'.format(df.shape))

    if not model_config['is_big']:
        # features from datetime
        df = transform_datetime_features(df)

        # categorical encoding
        for col_name, unique_values in model_config['categorical_values'].items():
            for unique_value in unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    # missing values
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif any(df.isnull()):
        df.fillna(value=df.mean(axis=0), inplace=True)

    number_columns = model_config['number_columns']
    id_columns = model_config['id_columns']
    datetime_columns = model_config['datetime_columns']

    if model_config['mode'] == MODE_regression \
            and len(datetime_columns) > 0 \
            and len(id_columns) > 0:

        print('Add shift columns for numeric')

        # def f_trans(x):
        #     for cn in number_columns:
        #         for i in range(1, 2 * 7 + 1):
        #             cn_shift = '{}_s{}'.format(cn, i)
        #             x[cn_shift] = x[cn].shift(i)
        #             x[cn_shift].fillna(-1, inplace=True)
        #     return x
        # def f_trans(x):
        #     x['number_23_s'] = x['number_23'].shift(-1).fillna(0)
        #
        #     return x
        #
        # df = df[id_columns + ['line_id'] + number_columns].groupby(id_columns).apply(f_trans)

    # filter columns
    used_columns = model_config['used_columns']

    # scale
    # X_scaled = model_config['scaler'].transform(df[used_columns])
    X_scaled = df[used_columns]

    model = model_config['model']
    if model_config['mode'] == MODE_regression:
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict_proba(X_scaled)[:, 1]

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    main(args)
