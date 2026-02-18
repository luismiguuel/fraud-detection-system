import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

print("🚀 Iniciando o processamento...")

# Carregar os dados de treino e teste
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Dados Carregados. Treino: {train.shape}, Teste: {test.shape}")

# Pré-processamento
train['Amount'] = np.log1p(train['Amount'])
test['Amount'] = np.log1p(test['Amount'])

# Padronizando a coluna 'Time' para ajudar o modelo a aprender melhor
sc = StandardScaler()
train['Time'] = sc.fit_transform(train['Time'].values.reshape(-1, 1))
test['Time'] = sc.transform(test['Time'].values.reshape(-1, 1))

# Separando as features e o target
cols_to_drop = ['id', 'Class']
X = train.drop(cols_to_drop, axis=1)
y = train['Class']

# Removendo 'id' do teste mas mantendo para a submissão final
X_test = test.drop(['id'], axis=1)

# Parâmetros otimizados para alta performance e dados desbalanceados
params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,      
    'boosting_type': 'gbdt',
    'random_state': SEED, 
    'learning_rate': 0.02,      
    'num_leaves': 64,           
    'max_depth': -1,
    'feature_fraction': 0.8,    
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_jobs': -1,               #
    'verbosity': -1
}


# Isso treina o modelo 5 vezes em partes diferentes dos dados para garantir robustez

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

print("\nIniciando Treinamento Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Criando o modelo
    model = lgb.LGBMClassifier(**params, n_estimators=2000)
    
    # Treinando com early stopping (para de treinar se não melhorar após 100 rodadas)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
    )
    
    # Previsão na parte de validação
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    # Pontuação
    score = roc_auc_score(y_val, val_pred)
    print(f"   -> Fold {fold+1} AUC: {score:.5f}")
    
    # Previsão no arquivo de teste final
    test_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits

# Resultado
final_auc = roc_auc_score(y, oof_preds)
print(f"\nAUC Média Geral (Validação Interna): {final_auc:.5f}")
print("-" * 30)

# Criação do arquivo de submissão
submission = pd.DataFrame({
    'id': test['id'],
    'Class': test_preds
})

submission.to_csv('submission.csv', index=False)
