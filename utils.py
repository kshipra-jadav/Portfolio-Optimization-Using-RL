import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    df = pd.read_csv('data/Quote-Equity-NETF-EQ-12-11-2023-to-12-11-2024.csv')
    df2 = pd.read_csv('data/Quote-Equity-ITBEES-EQ-12-11-2023-to-12-11-2024.csv')
    df3 = pd.read_csv('data/Quote-Equity-KOTAKBKETF-EQ-12-11-2023-to-12-11-2024.csv')
    df4 = pd.read_csv('data/Quote-Equity-LICNFNHGP-EQ-12-11-2023-to-12-11-2024.csv')
    df5 = pd.read_csv('data/Quote-Equity-NEXT50IETF-EQ-12-11-2023-to-12-11-2024.csv')

    df_concat = pd.concat([df['close '], df2['close '], df3['close '], df4['close '], df5['close ']], axis=1)
    df_concat.columns = ['close', 'close2', 'close3', 'close4', 'close5']

    return df_concat

