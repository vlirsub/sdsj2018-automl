import datetime


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
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        # День недели
        df['dt_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        # Будни или выходной день
        #df['dt_is_work_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday() >= 5)
        # Квартал
        #df['dt_quarter_{}'.format(col_name)] = df[col_name].apply(lambda x: x.quarter)
        # Месяц
        df['dt_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        # День года
        #df['dt_dayofyear_{}'.format(col_name)] = df[col_name].apply(lambda x: x.dayofyear)
        # Неделя года
        #df['dt_weekofyear_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekofyear)
        # День
        df['dt_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        # Час
        # df['dt_number_dt_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        # Час недели
        # df['dt_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        # Минута дня
        # df['dt_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
    return df
