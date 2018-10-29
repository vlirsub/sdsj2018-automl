import datetime
import pandas as pd
import numpy as np


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


def transform_datetime_features(df):
    datetime_columns = [col_name for col_name in df.columns if col_name.startswith('datetime')]
    for col_name in datetime_columns:
        # df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        # df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        # df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        # df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        # df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        # df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        # df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)

        df[col_name] = pd.to_datetime(df[col_name])
        df['number_day_{}'.format(col_name)] = df[col_name].dt.weekday.astype(np.float16)
        df['number_month_{}'.format(col_name)] = df[col_name].dt.month.astype(np.float16)
        df['number_day_{}'.format(col_name)] = df[col_name].dt.day.astype(np.float16)
    return df


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# eof
