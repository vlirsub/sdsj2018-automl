import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

MODE_REGRESSION = 'regression'
MODE_CLASSIFICATION = 'classification'

def main(args):
    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    mode = model_config['mode']
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
    datetime_columns = model_config['datetime_columns']
    id_columns = model_config['id_columns']

    if len(id_columns) > 0 and len(datetime_columns) > 0 and mode == MODE_REGRESSION:
        # check_3
        def f_trans(x):
            for cn in number_columns:
                x['{}_s{}'.format(cn, -1)] = x[cn].shift(-1).fillna(0)

            return x

        df = df[id_columns + ['line_id'] + number_columns].groupby(id_columns).apply(f_trans)

    if 3 <= len(datetime_columns) <= 10:
        # check_4
        print('Add delta datetime columns')
        for cn in datetime_columns:
            df[cn] = pd.to_datetime(df[cn])
        import itertools
        for c1, c2 in list(itertools.combinations(datetime_columns, 2)):
            df['number_{}_{}'.format(c1, c2)] = (df[c1] - df[c2]).dt.days

    # filter columns
    used_columns = model_config['used_columns']

    # scale
    #X_scaled = model_config['scaler'].transform(df[used_columns])
    X_scaled = df[used_columns]

    model = model_config['model']
    if mode == MODE_REGRESSION:
        df['prediction'] = model.predict(X_scaled)
    elif mode == MODE_CLASSIFICATION:
        df['prediction'] = model.predict_proba(X_scaled)[:, 1]
    else:
        raise Exception('Ошибочный режим {}'.format(mode))

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


