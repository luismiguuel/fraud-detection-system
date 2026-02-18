import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

SEED = 42

def train_ensemble(X, y, X_test):
    """Treina LightGBM e XGBoost com Cross-Validation e salva os artefatos."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Arrays para guardar previsões
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    test_pred_lgb = np.zeros(len(X_test))
    test_pred_xgb = np.zeros(len(X_test))
    
    # LightGBM
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'is_unbalance': True,
        'verbosity': -1, 'seed': SEED, 'learning_rate': 0.01,
        'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    # Treino simplificado
    final_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
    final_lgb.fit(X, y)
    joblib.dump(final_lgb, 'models/lgbm_model.pkl')
    
    # XGBoost
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'scale_pos_weight': ratio, 'learning_rate': 0.02,
        'max_depth': 4, 'seed': SEED, 'n_jobs': -1
    }
    
    final_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=1000)
    final_xgb.fit(X, y)
    joblib.dump(final_xgb, 'models/xgboost_model.pkl')

    # Cross-Validation    
    for train_idx, val_idx in folds.split(X, y):
        X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
        X_val_cv, y_val_cv = X.iloc[val_idx], y.iloc[val_idx]
        
        lgb_fold = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
        lgb_fold.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_val_cv)[:, 1]
        
        xgb_fold = xgb.XGBClassifier(**xgb_params, n_estimators=1000, early_stopping_rounds=50)
        xgb_fold.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_val_cv)[:, 1]

        # Acumula previsões de teste
        test_pred_lgb += lgb_fold.predict_proba(X_test)[:, 1] / 5
        test_pred_xgb += xgb_fold.predict_proba(X_test)[:, 1] / 5

    # Calculando score 
    ensemble_oof = (oof_lgb * 0.5) + (oof_xgb * 0.5)
    final_score = roc_auc_score(y, ensemble_oof)
    print(f"AUC Score Global (Ensemble): {final_score:.5f}")
    
    # Gera a previsão final combinada
    final_test_preds = (test_pred_lgb * 0.5) + (test_pred_xgb * 0.5)
    
    return final_test_preds