import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import os

SEED = 42

def train_ensemble(X, y, X_test):
    """Treina LightGBM e XGBoost tunados com CV Temporal e salva os artefatos."""
    
    folds = TimeSeriesSplit(n_splits=5)
    
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    test_pred_lgb = np.zeros(len(X_test))
    test_pred_xgb = np.zeros(len(X_test))
    
    val_indices = []
    
    # Parâmetros otimizados
    lgb_params = {
        'objective': 'binary', 
        'metric': 'auc', 
        'verbosity': -1, 
        'boosting_type': 'gbdt',
        'seed': SEED,
        'is_unbalance': False,
        'learning_rate': 0.014091355706800627,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 89,
        'subsample': 0.8864275131984432,
        'colsample_bytree': 0.6704516247725975,
    }
    
    final_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
    final_lgb.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_lgb, 'models/lgbm_tuned.pkl')
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    xgb_params = {
        'objective': 'binary:logistic', 
        'eval_metric': 'auc',
        'seed': SEED, 
        'n_jobs': -1,
        'scale_pos_weight': ratio, 
        'learning_rate': 0.0489604093407068,
        'max_depth': 4,
        'min_child_weight': 4,
        'gamma': 0.12214224426073485,
        'subsample': 0.8887598020857413,
        'colsample_bytree': 0.5463025712871026,
    }
    
    final_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=1500)
    final_xgb.fit(X, y)
    joblib.dump(final_xgb, 'models/xgboost_tuned.pkl')

    # Cross-validation
    for train_idx, val_idx in folds.split(X, y):
        val_indices.extend(val_idx)
        
        X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
        X_val_cv, y_val_cv = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM Fold
        lgb_fold = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
        lgb_fold.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_val_cv)[:, 1]
        
        # XGBoost Fold
        xgb_fold = xgb.XGBClassifier(**xgb_params, n_estimators=1500, early_stopping_rounds=50)
        xgb_fold.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_val_cv)[:, 1]

        # Acumula as previsões do Teste Invisível (TTA)
        test_pred_lgb += lgb_fold.predict_proba(X_test)[:, 1] / 5
        test_pred_xgb += xgb_fold.predict_proba(X_test)[:, 1] / 5

    # Avalia o ensemble OOF (50% LGBM + 50% XGBoost)
    ensemble_oof = (oof_lgb[val_indices] * 0.5) + (oof_xgb[val_indices] * 0.5)
    final_score = roc_auc_score(y.iloc[val_indices], ensemble_oof)
    
    print(f"AUC Score Global (Tuned Ensemble): {final_score:.5f}")
    
    # Combina as previsões do teste final (50% LGBM + 50% XGBoost)
    final_test_preds = (test_pred_lgb * 0.5) + (test_pred_xgb * 0.5)
    
    return final_test_preds