import pandas as pd
import glob
import os
from datetime import datetime, timedelta

def compute_corr_mat(df,time_interval):
    # Assurez-vous que la colonne timestamp est de type datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Calculer les fonctionnalités dérivées
    df['spread'] = df['ask_price'] - df['bid_price']
    df['volume_imbalance'] = df['bid_volume'] - df['ask_volume']

    # Calculer les retours (returns) pour les prix, volumes, spreads, et volume imbalance
    df['price_return'] = df.groupby('index_id')['price'].pct_change()
    df['volume_return'] = df.groupby('index_id')['volume'].pct_change()
    df['spread_return'] = df['spread'].pct_change()
    df['volume_imbalance_return'] = df['volume_imbalance'].pct_change()

    # Agréger les données par intervalle de temps
    aggregated_df = df.resample(time_interval).agg({
        'price_return': 'mean',
        'volume_return': 'mean',
        'spread_return': 'mean',
        'volume_imbalance_return': 'mean'
    }).dropna()

    # Normaliser les données
    aggregated_df = aggregated_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    # Calculer la matrice de corrélation
    correlation_matrix = aggregated_df.corr()

    return correlation_matrix


import pandas as pd
import numpy as np


def process_index_dataframe(df, start_date, end_date, time_interval='15T'):
    """
    Extract feature vector (FV) for an index DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with columns ['normal_date', 'trade-price', 'trade-volume', 'bid-price', 'ask-price', 'bid-volume', 'ask-volume']
        time_interval (str): Time interval for resampling (e.g., '5T', '15T', '30T')
    Returns:
        pd.DataFrame: Feature Vector DataFrame with columns [ΔPrice, ΔVolume, ΔSpread, ΔVolImb]
    """
    # Ensure normal_date is datetime
    df.reset_index(inplace=True)
    df['normal_date'] = pd.to_datetime(df['normal_date'])
    df.set_index('normal_date', inplace=True)

    # Calculate derived features
    df['spread'] = df['ask-price'] - df['bid-price']
    df['volume_imbalance'] = df['bid-volume'] - df['ask-volume']

    # Aggregate data by time interval
    df_resampled = df.resample(time_interval).agg({
        'trade-price': 'mean',
        'trade-volume': 'sum',
        'spread': 'mean',
        'volume_imbalance': 'mean'
    }).dropna()
    full_index = pd.date_range(start=start_date, end=end_date, freq=time_interval)
    df_resampled = df_resampled.reindex(full_index)

    # Calculate percentage changes (Δ features)
    df_resampled['ΔPrice'] = df_resampled['trade-price'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df_resampled['ΔVolume'] = df_resampled['trade-volume'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df_resampled['ΔSpread'] = df_resampled['spread'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df_resampled['ΔVolImb'] = df_resampled['volume_imbalance'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Return only the feature vector columns
    fv_df = df_resampled[['ΔPrice', 'ΔVolume', 'ΔSpread', 'ΔVolImb']]

    return fv_df



def merge_index_dataframes(df1, df2):
    """
    Merge two DataFrames for the same index based on 'normal_date'.

    Args:
        df1 (pd.DataFrame): First DataFrame with trade data (trade-price, trade-volume).
        df2 (pd.DataFrame): Second DataFrame with bid-ask data (bid-price, bid-volume, ask-price, ask-volume).

    Returns:
        pd.DataFrame: Merged DataFrame with all relevant features.
    """
    # Ensure normal_date is in datetime format for both DataFrames
    df1['normal_date'] = pd.to_datetime(df1['normal_date'])
    df2['normal_date'] = pd.to_datetime(df2['normal_date'])

    # Set normal_date as the index for both DataFrames
    df1 = df1.set_index('normal_date', inplace=False)
    df2 = df2.set_index('normal_date', inplace=False)

    # Merge DataFrames on 'normal_date' using an outer join to preserve all data
    df_merged = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')

    # Sort by timestamp
    df_merged.sort_index(inplace=True)
    # Fill or handle missing values if necessary
    # df_merged.fillna(method='ffill', inplace=True)  # Forward fill for time series data
    df_merged[['bid-price', 'bid-volume', 'ask-price', 'ask-volume', 'trade-price','trade-volume']] = df_merged[['bid-price', 'bid-volume', 'ask-price', 'ask-volume', 'trade-price','trade-volume']].fillna(method='ffill')

    return df_merged


def convert_xltime_to_date(df, xltime_column='xltime'):
    """
    Converts Excel serial date-time to normal datetime format.

    Args:
        df (pd.DataFrame): The input DataFrame with Excel serial dates.
        xltime_column (str): The column containing Excel serial date-time values.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'normal_date' column.
    """
    # Define Excel's starting reference date
    excel_start_date = datetime(1899, 12, 30)  # Excel incorrectly considers 1900 as a leap year

    # Convert Excel serial date to datetime
    df['normal_date'] = df[xltime_column].apply(
        lambda x: excel_start_date + timedelta(days=float(x))
    )

    return df


def pipeline_trade_bbo(df_bbo,df_trade,time_interval,start_date,end_date):
    df_bbo = convert_xltime_to_date(df_bbo)
    df_trade = convert_xltime_to_date(df_trade)
    df_bbo.reset_index(inplace=True)
    df_bbo = df_bbo[['normal_date','bid-price','bid-volume','ask-price','ask-volume']]
    df_trade = df_trade[['normal_date','trade-price','trade-volume']]
    df_final = merge_index_dataframes(df_bbo,df_trade)
    return process_index_dataframe(df_final, start_date, end_date, time_interval=time_interval)

import pandas as pd
from functools import reduce

def merge_multiple_dataframes(dfs):
    """
    Merge a list of DataFrames on their index using an outer join.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to merge.

    Returns:
        pd.DataFrame: A single merged DataFrame.
    """


    # Perform the merge using reduce
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how='outer'
    ), dfs)

    # Sort by index to ensure chronological order
    merged_df.sort_index(inplace=True)


    return merged_df


def corr_mat(index_to_files, start_date, end_date, time_interval):
    sd = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    ed = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    per_index_dfs = []
    for index in index_to_files:
        # Filter trade files by date
        files_trade = glob.glob(os.path.join(index_to_files[index]['trade'], '*.csv.gz'))
        filtered_trade_files = [
            file for file in files_trade
            if sd <= datetime.strptime('-'.join(os.path.basename(file).split('-')[:3]), '%Y-%m-%d') <= ed
        ]
        combined_df_trade = pd.concat(
            (pd.read_csv(file, compression='gzip') for file in filtered_trade_files), ignore_index=True
        )

        files_bbo = glob.glob(os.path.join(index_to_files[index]['bbo'], '*.csv.gz'))
        filtered_bbo_files = [
            file for file in files_bbo
            if sd <= datetime.strptime('-'.join(os.path.basename(file).split('-')[:3]), '%Y-%m-%d') <= ed
        ]
        combined_df_bbo = pd.concat(
            (pd.read_csv(file, compression='gzip') for file in filtered_bbo_files), ignore_index=True
        )

        feature_vec_idx = pipeline_trade_bbo(combined_df_bbo, combined_df_trade, time_interval, start_date, end_date)
        per_index_dfs.append(feature_vec_idx)

    return merge_multiple_dataframes(per_index_dfs).T.corr()