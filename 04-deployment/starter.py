#!/usr/bin/env python
# coding:

from pathlib import Path
import argparse

import pickle
import pandas as pd


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def apply_model(year, month):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/'\
                    f'yellow_tripdata_{year:04d}-{month:02d}.parquet')

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    dv, model = load_model()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    taxi_type = "yellow"

    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    path1 = Path(f'output/')
    path2 = Path(f'output/{taxi_type}')

    try:
        path1.mkdir()
        path2.mkdir()
    except FileExistsError:
        pass

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f'mean predicted duration: {df_result.mean()}')
    print(f'result saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="An application to predict taxi rides.")
    parser.add_argument('year', help='The year')
    parser.add_argument('month', help='The month')

    args = parser.parse_args()
    if args.year and args.month:
        year = int(args.year)
        month = int(args.month)
        apply_model(year,month)

if __name__ == '__main__':
    main()