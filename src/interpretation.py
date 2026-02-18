import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def explain_model(model, X_train, model_name="model"):
    """
    Gera gráficos de interpretabilidade usando SHAP.
    Mostra quais features mais impactaram nas decisões do modelo.
    """
    print(f"\nGerando explicações SHAP para {model_name}...")
    
    # Cria pasta para salvar os gráficos se não existir
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Para Tree-based models, é usado TreeExplainer
    X_sample = X_train.sample(n=1000, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Pega o índice 1 (classe positiva/fraude) se for lista
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Gráfico de Summary 
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"Impacto das Features - {model_name}")
    plt.tight_layout()
    plt.savefig(f'reports/shap_summary_{model_name}.png')
    plt.close()
    
    print(f"Gráfico salvo em 'reports/shap_summary_{model_name}.png'")