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
    print('is_null_target {}'.format(model_config['is_null_target']))

    if model_config['is_work']:
        if model_config['is_null_target']:
            # filter columns
            used_columns = model_config['used_columns']

            X_test = df[used_columns]

            model_c = model_config['model_c']
            model_r = model_config['model_r']

            # Проноз c
            prediction_c = model_c.predict(X_test)
            # Проноз r
            prediction_r = model_r.predict(X_test)

            prediction_r[prediction_c == 0] = 0
            df['prediction'] = prediction_r
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


