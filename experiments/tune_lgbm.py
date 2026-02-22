import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Importa o pré-processamento 
from src.preprocessing import load_data, prepare_data

SEED = 42

def objective(trial):
    # Carrega e prepara os dados
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    X, y, _ = prepare_data(train_df, test_df)
    
    # Calcula a proporção exata de desbalanceamento do target para o LightGBM
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': SEED,
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
    }

    # Validação temporal rigorosa 
    folds = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(X))
    val_indices = []

    for train_idx, val_idx in folds.split(X, y):
        val_indices.extend(val_idx)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param, n_estimators=1000)
        
        # Treina com Early Stopping para não perder tempo com árvores ruins
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Calcula o score apenas nos dados validados 
    score = roc_auc_score(y.iloc[val_indices], oof_preds[val_indices])
    return score

if __name__ == "__main__":
    print("Iniciando a caçada aos melhores hiperparâmetros para o LightGBM...")
    # Optuna tenta MAXIMIZAR o AUC
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print("\n" + "="*50)
    print(f"Melhor AUC Local Encontrado: {study.best_value:.5f}")
    print("Melhores Parâmetros:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("="*50)