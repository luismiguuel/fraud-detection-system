import pandas as pd
import numpy as np
import joblib
from src.preprocessing import load_data, prepare_data
from src.training import train_ensemble
from src.interpretation import explain_model

SEED = 42
np.random.seed(SEED)

def main():
    print("Detectando Fraudes")
    
    # Carregamento
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    
    # Pré-processamento
    X, y, X_test = prepare_data(train_df, test_df)
    
    print(f"Features processadas: {X.shape[1]} colunas.")
    
    # Treinamento e Validação
    final_predictions = train_ensemble(X, y, X_test)

    # Interpretabilidade e geração de gráficos
    lgbm_model = joblib.load('models/lgbm_model.pkl')
    xgbm_model = joblib.load('models/xgboost_model.pkl')

    explain_model(lgbm_model, X, model_name="LGBM")
    explain_model(xgbm_model, X, model_name="XGBM")
    
    # Geração de Submissão
    print("Gerando arquivo de submissão...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Class': final_predictions
    })
    
    submission.to_csv('submission.csv', index=False)

main()