import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

def main(args):
    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Test Dataset read, shape {}'.format(df.shape))
    print('Dataset memory usage {:.3} MB'.format(df.memory_usage().sum() / 1024 / 1024))
    print('is_work {}'.format(model_config['is_work']))

    if model_config['is_work']:
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

        # filter columns
        used_columns = model_config['used_columns']

        # scale
        #X_scaled = model_config['scaler'].transform(df[used_columns])
        X_scaled = df[used_columns]

        model = model_config['model']
        if model_config['mode'] == 'regression':
            df['prediction'] = model.predict(X_scaled)
        elif model_config['mode'] == 'classification':
            df['prediction'] = model.predict_proba(X_scaled)[:, 1]
    else:
        df['prediction'] = 0

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


