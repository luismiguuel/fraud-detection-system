import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def prepare_data(train_df, test_df):
    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # 1. A sua feature original de ouro
    train_proc['Hour'] = train_proc['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    test_proc['Hour'] = test_proc['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)

    # 2. Scalers blindados contra vazamento de dados
    rs_amount = RobustScaler()
    train_proc['Amount_Scaled'] = rs_amount.fit_transform(train_proc[['Amount']])
    test_proc['Amount_Scaled'] = rs_amount.transform(test_proc[['Amount']])

    rs_time = RobustScaler()
    train_proc['Time_Scaled'] = rs_time.fit_transform(train_proc[['Time']])
    test_proc['Time_Scaled'] = rs_time.fit_transform(test_proc[['Time']])

    # 3. Limpeza
    train_proc.drop(['Time', 'Amount'], axis=1, inplace=True)
    test_proc.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = train_proc.drop(['id', 'Class'], axis=1)
    y = train_proc['Class']
    X_test = test_proc.drop(['id'], axis=1)

    return X, y, X_test