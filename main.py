import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

SEED = 42
np.random.seed(SEED)

print("Iniciando processamento avançado...")

# Carregar Dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Análise exploratória 
train['Hour'] = train['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
test['Hour'] = test['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)

# Normalizando a coluna Amount
rs = RobustScaler()
train['Amount'] = rs.fit_transform(train['Amount'].values.reshape(-1, 1))
test['Amount'] = rs.transform(test['Amount'].values.reshape(-1, 1))

# Normalizando o tempo
train['Time'] = rs.fit_transform(train['Time'].values.reshape(-1, 1))
test['Time'] = rs.transform(test['Time'].values.reshape(-1, 1))

# Preparando x e y
X = train.drop(['id', 'Class'], axis=1)
y = train['Class']
X_test = test.drop(['id'], axis=1)

print(f"Features criadas. Colunas atuais: {X.columns.tolist()}")

# Definição dos Modelos:

# MODELO A: LightGBM 
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,
    'verbosity': -1,
    'seed': SEED,
    'learning_rate': 0.01,    
    'num_leaves': 31,         
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,       
    'reg_lambda': 0.1
}

# MODELO B: XGBoost
# Calculando peso para classes desbalanceadas
ratio = float(np.sum(y == 0)) / np.sum(y == 1)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': ratio, 
    'learning_rate': 0.02,
    'max_depth': 4,            
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': SEED,
    'n_jobs': -1,
    'enable_categorical': False
}

# Estratégia de treinamento
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Arrays para guardar as previsões
oof_lgb = np.zeros(len(X))
test_pred_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_pred_xgb = np.zeros(len(X_test))

print("\nIniciando Treinamento Híbrido (Isso pode demorar um pouco)...")

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=5000)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_pred_lgb += lgb_model.predict_proba(X_test)[:, 1] / folds.n_splits
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=5000, early_stopping_rounds=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_pred_xgb += xgb_model.predict_proba(X_test)[:, 1] / folds.n_splits
    
    # Scores individuais
    score_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    score_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    
    # Score combinado
    ensemble_pred = (oof_lgb[val_idx] * 0.5) + (oof_xgb[val_idx] * 0.5)
    score_ens = roc_auc_score(y_val, ensemble_pred)
    
    print(f"Fold {fold+1} >> LGB: {score_lgb:.5f} | XGB: {score_xgb:.5f} | COMBO: {score_ens:.5f}")

# Avaliação final
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_ensemble = roc_auc_score(y, (oof_lgb * 0.5 + oof_xgb * 0.5))

print("-" * 40)
print(f"AUC Final LightGBM: {auc_lgb:.5f}")
print(f"AUC Final XGBoost:  {auc_xgb:.5f}")
print(f"AUC Final ENSEMBLE: {auc_ensemble:.5f}")

final_preds = (test_pred_lgb * 0.5) + (test_pred_xgb * 0.5)

submission = pd.DataFrame({
    'id': test['id'],
    'Class': final_preds
})

submission.to_csv('submission_ensemble.csv', index=False)
