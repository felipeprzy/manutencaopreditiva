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



# ==============================================================================
# FASE 3: ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS (FEATURE IMPORTANCE)
# ==============================================================================

print("\n--- Iniciando Fase 3: Cálculo da Importância das Features ---")

# Comentário Técnico: Extração dos scores de importância calculados pelas árvores do Random Forest.
# Permite identificar quais sensores ou parâmetros operacionais mais influenciam o risco de falha.
importancias = modelo_rf.feature_importances_
nomes_features = X.columns

# Comentário Técnico: Organização dos dados em um DataFrame para facilitar a visualização e ordenação.
df_importancia = pd.DataFrame({
    'Variavel': nomes_features,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

print("Ranking de Impacto das Variáveis na Manutenção Preditiva:")
print(df_importancia.to_string(index=False))

print("\nFase 3 concluída com sucesso: Importância das features mapeada.")


# ==============================================================================
# FASE 4: VALIDAÇÃO CRUZADA, TESTES AVANÇADOS E VISUALIZAÇÃO GRÁFICA
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

print("\n--- Iniciando Fase 4: Validação Avançada e Gráficos de Desempenho ---")

# Comentário Técnico: Aplicação de Validação Cruzada (5-fold cross-validation).
# Garante a robustez do modelo avaliando sua estabilidade em diferentes partições dos dados de treino.
scores_cv = cross_val_score(modelo_rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Acurácias individuais (5-fold): {scores_cv}")
print(f"Média da Validação Cruzada: {scores_cv.mean() * 100:.2f}% (+/- {scores_cv.std() * 100:.2f}%)")

# Comentário Técnico: Cálculo da Curva ROC e da Área Sob a Curva (AUC) para medir a capacidade discriminatória do classificador.
y_prob = modelo_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"Área sob a Curva ROC (AUC): {roc_auc:.4f}")

# Comentário Técnico: Geração e salvamento do gráfico da Matriz de Confusão em formato de mapa de calor (Heatmap).
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Falha'], yticklabels=['Normal', 'Falha'])
plt.title('Matriz de Confusão - Manutenção Preditiva')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.savefig('matriz_confusao.png')
plt.close()

# Comentário Técnico: Geração e salvamento do gráfico da Curva ROC para análise visual da taxa de acerto vs falsos alarmes.
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Modelo Preditivo de Falhas')
plt.legend(loc="lower right")
plt.savefig('curva_roc.png')
plt.close()

print("Gráficos analíticos gerados e salvos com sucesso ('matriz_confusao.png', 'curva_roc.png').")
print("\nFase 4 concluída: Validação estatística e visualizações integradas ao pipeline.")


