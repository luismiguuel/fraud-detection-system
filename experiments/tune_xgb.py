import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from src.preprocessing import load_data, prepare_data

SEED = 42

def objective(trial):
    # Carrega os dados
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    X, y, _ = prepare_data(train_df, test_df)
    
    # Calcula a proporção exata de desbalanceamento do target
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    # O espaço de busca estratégico para o XGBoost
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': SEED,
        'n_jobs': -1,
        'scale_pos_weight': ratio, 
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
    }

    # Validação temporal 
    folds = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(X))
    val_indices = []

    for train_idx, val_idx in folds.split(X, y):
        val_indices.extend(val_idx)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Treina com Early Stopping para evitar overfitting e economizar tempo
        model = xgb.XGBClassifier(**param, n_estimators=1000, early_stopping_rounds=30)
        
        # Treina o modelo
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Score OOF Temporal rigoroso
    score = roc_auc_score(y.iloc[val_indices], oof_preds[val_indices])
    return score

if __name__ == "__main__":
    print("Iniciando a caçada aos melhores hiperparâmetros para o XGBoost...")
    # Optuna tenta MAXIMIZAR o AUC
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print("\n" + "="*50)
    print(f"Melhor AUC Local Encontrado: {study.best_value:.5f}")
    print("Melhores Parâmetros:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("="*50)