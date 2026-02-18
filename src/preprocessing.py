import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def load_data(train_path, test_path):
    """Carrega os datasets brutos"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def feature_engineering(df):
    """Aplica transformações nos dados"""
    df_out = df.copy()
    
    # Hora do dia
    # Transforma segundos em horas (0 a 23)
    df_out['Hour'] = df_out['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    
    # Normalização para 'Amount' e 'Time'
    rs = RobustScaler()
    df_out['Amount_Scaled'] = rs.fit_transform(df_out['Amount'].values.reshape(-1, 1))
    df_out['Time_Scaled'] = rs.fit_transform(df_out['Time'].values.reshape(-1, 1))
    
    # Remove colunas originais
    df_out.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    return df_out

def prepare_data(train_df, test_df):
    """Orquestra o pré-processamento."""    
    train_proc = feature_engineering(train_df)
    test_proc = feature_engineering(test_df)
    
    # Separação x e y
    X = train_proc.drop(['id', 'Class'], axis=1)
    y = train_proc['Class']
    X_test = test_proc.drop(['id'], axis=1)
    
    return X, y, X_test