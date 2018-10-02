# -*- coding: utf-8 -*-
# Запуск скриптов для соревнования сбера

import sys
import os
import datetime as dt
from collections import namedtuple
import time

import train
import predict

from sklearn.metrics import roc_auc_score, mean_squared_error
import pandas as pd

# "train_classification": "python train.py --mode classification --train-csv {train_csv} --model-dir {model_dir}",
# "train_regression": "python train.py --mode regression --train-csv {train_csv} --model-dir {model_dir}",
# "predict": "python predict.py --test-csv {test_csv} --prediction-csv {prediction_csv} --model-dir {model_dir}"

# Обучение
PyTrainArgs = namedtuple('PyTrainArgs', 'mode,train_csv,model_dir')
# Прогноз
PyPredictArgs = namedtuple('PyPredictArgs', 'test_csv,prediction_csv,model_dir')
# Результат
PyResult = namedtuple('PyResult', 'name,time,metric')

MODE_classification = r'classification'
MODE_regression = r'regression'

# Директория для записи
# MODEL_DIR = r'q:\sdsj2018-automl-master\examples\baseline\.local'
MODEL_DIR = r'.local'

# Директория с данными
DATA_DIR = r'm:\tmp\SB'


def get_model(c):
    """Определение типа модели"""
    if c == MODE_regression[0]:
        return MODE_regression
    elif c == MODE_classification[0]:
        return MODE_classification
    else:
        raise Exception('Не ожиданный тип модели "{}"'.format(d[-1]))


def main():
    # Результат
    Result = list()

    dirs = os.listdir(DATA_DIR)
    for d in dirs:
        fd = os.path.join(DATA_DIR, d)
        if os.path.isdir(fd):
            start_time = time.time()
            print('========== Train ==========')
            print(fd)
            # Определяем тип модели
            mode = get_model(d[-1])

            # Данные для обучени
            train_csv = os.path.join(fd, r'train.csv')
            model_dir = os.path.join(MODEL_DIR, d)
            os.makedirs(model_dir, exist_ok=True)
            args = PyTrainArgs(mode, train_csv, model_dir)
            # print(args)

            # Обучение
            train.main(args)

            print('========== Predict ==========')
            # Данные для тестирования
            test_csv = os.path.join(fd, r'test.csv')
            # Правильные ответы
            target_csv = os.path.join(fd, r'test-target.csv')

            # Результаты прогноза
            prediction_csv = os.path.join(model_dir, r'prediction.csv')

            args = PyPredictArgs(test_csv, prediction_csv, model_dir)
            # print(args)
            # Прогноз
            predict.main(args)

            print('Считаем метрики решения')
            if os.path.exists(prediction_csv):
                y_true = pd.read_csv(target_csv)
                y_score = pd.read_csv(prediction_csv)
                y = pd.merge(y_true, y_score, how='left', on='line_id')

                if mode == MODE_classification:
                    # roc_auc
                    metric = roc_auc_score(y['target'], y['prediction'])
                    print('roc auc: {:.4}'.format(metric))
                elif mode == MODE_regression:
                    # RMSE
                    metric = mean_squared_error(y['target'], y['prediction'])
                    print('RMSE: {:.4}'.format(metric))
                else:
                    raise Exception('Не ожиданный тип модели')
            else:
                print('Не найден файл с ответами', file=sys.stderr)
                metric = 0

            Result.append(PyResult(d, time.time() - start_time, metric))
            # Для проверки 1 датасета
            break

    Result = pd.DataFrame(Result)
    # Результирующий файл
    file_out = os.path.join(MODEL_DIR, 'result_{}.csv'.format(dt.datetime.now().strftime('%Y-%m-%d %H.%M.%S')))
    Result.to_csv(file_out, index=False, float_format='%.4f')
    print('Результат сохранен в {}'.format(file_out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
