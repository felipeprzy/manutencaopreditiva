import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
​print("Iniciando a Fase 1: Carregamento e Pré-processamento dos Dados de Manutenção Preditiva...")
​np.random.seed(42)
n_amostras = 1000
​dados = {
'tempo_operacao_horas': np.random.randint(100, 5000, size=n_amostras),
'temperatura_media': np.random.uniform(50.0, 110.0, size=n_amostras),
'vibracao_mm_s': np.random.uniform(0.5, 8.5, size=n_amostras),
'pressao_oleo': np.random.uniform(20.0, 60.0, size=n_amostras)
}
​df = pd.DataFrame(dados)
​df['falha_iminente'] = (
(df['temperatura_media'] > 95.0) |
(df['vibracao_mm_s'] > 6.5) |
(df['tempo_operacao_horas'] > 4200)
).astype(int)
​print(f"Base de dados gerada com sucesso! Total de registros: {len(df)}")
print(df.head())
​X = df[['tempo_operacao_horas', 'temperatura_media', 'vibracao_mm_s', 'pressao_oleo']]
y = df['falha_iminente']
​X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
​scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
​print("Fase 1 concluída com sucesso: Dados normalizados e prontos para o treinamento dos modelos de machine learning.")


# ==============================================================================
# FASE 2: TREINAMENTO E AVALIAÇÃO DO MODELO PREDITIVO (RANDOM FOREST)
# ==============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n--- Iniciando Fase 2: Treinamento do Modelo Random Forest ---")

# Comentário Técnico: Instanciação do classificador Random Forest com 100 árvores de decisão.
# O n_estimators=100 garante um bom equilíbrio entre capacidade de generalização e custo computacional.
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Comentário Técnico: Ajuste (treinamento) do modelo utilizando os dados normalizados da Fase 1.
modelo_rf.fit(X_train_scaled, y_train)

# Comentário Técnico: Geração das predições para o conjunto de teste (dados não vistos durante o treino).
y_pred = modelo_rf.predict(X_test_scaled)

# ==============================================================================
# AVALIAÇÃO DE DESEMPENHO E MÉTRICAS
# ==============================================================================

# Comentário Técnico: Cálculo e exibição das métricas de precisão, recall e f1-score.
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia Geral do Modelo: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Falha Iminente (1)']))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nFase 2 concluída: Modelo treinado e avaliado com sucesso.")

