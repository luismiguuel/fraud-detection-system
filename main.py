import pandas as pd
import numpy as np
from src.preprocessing import load_data, prepare_data
from src.training import train_ensemble

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
    
    # Geração de Submissão
    print("Gerando arquivo de submissão...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Class': final_predictions
    })
    
    submission.to_csv('submission.csv', index=False)

main()