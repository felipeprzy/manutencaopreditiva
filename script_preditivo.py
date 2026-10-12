# =================================================================ar
# PROJETO: PIPELINE PREDITIVO - INDÚSTRIA 4.0
# FASE 1: ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. Carregamento da Base de Dados Real
# ------------------------------------------------------------------------------
# Justificativa técnica: Utiliza-se caminho relativo para garantir a 
# reprodutibilidade do script em qualquer ambiente de testes (exigência do edital).
# Substitua 'nome_do_seu_arquivo.csv' pelo nome exato do arquivo CSV enviado ao repositório.
CAMINHO_DADOS = 'nome_do_seu_arquivo.csv'

try:
    df = pd.read_csv(CAMINHO_DADOS)
    print("Sucesso: Base de dados real carregada com êxito.")
except Exception as e:
    print(f"Erro ao carregar o arquivo de dados: {e}")

# ------------------------------------------------------------------------------
# 2. Análise Estrutural e Descritiva Inicial
# ------------------------------------------------------------------------------
# Dimensões do Dataset: Número de linhas (instâncias) e colunas (variáveis)
print("\n--- DIMENSÕES DO DATASET ---")
linhas, colunas = df.shape
print(f"Número total de linhas (registros): {linhas}")
print(f"Número total de colunas (variáveis): {colunas}")

# Tipos de dados de cada variável do processo fabril
print("\n--- TIPOS DE DADOS DAS VARIÁVEIS ---")
print(df.dtypes)

# Resumo estatístico descritivo das colunas numéricas (.describe())
# Fornece média, desvio padrão, mínimo, quartis e valor máximo para identificar assimetrias
print("\n--- RESUMO ESTATÍSTICO DESCRITIVO (.describe()) ---")
display(df.describe())

# ------------------------------------------------------------------------------
# 3. Geração dos Gráficos Analíticos Obrigatórios
# ------------------------------------------------------------------------------

# Gráfico 1: Histograma de distribuição de uma variável preditora principal
# Objetivo: Observar a frequência e o comportamento de dispersão dos dados dos sensores
plt.figure(figsize=(8, 5))
coluna_alvo_hist = df.select_dtypes(include=[np.number]).columns[0] # Seleciona a primeira coluna numérica
sns.histplot(df[coluna_alvo_hist], kde=True, color='blue')
plt.title(f'Fase 1 - Histograma de Distribuição: {coluna_alvo_hist}')
plt.xlabel(coluna_alvo_hist)
plt.ylabel('Frequência')
plt.savefig('hist_distribuicao_variavel.png')
plt.close()

# Gráfico 2: Gráfico de barras da variável alvo (Falha)
# Objetivo: Comprovar visualmente a taxa de desbalanceamento (Falha = 1 vs Normal = 0)
plt.figure(figsize=(6, 4))
coluna_alvo_binaria = df.columns[-1] # Assume-se que a última coluna seja o alvo 'falha'
sns.countplot(x=coluna_alvo_binaria, data=df, palette='Set2')
plt.title('Fase 1 - Taxa de Desbalanceamento da Variável Alvo (Falha)')
plt.xlabel('Classe (0 = Normal, 1 = Falha)')
plt.ylabel('Contagem de Registros')
plt.savefig('barras_desbalanceamento_alvo.png')
plt.close()

# Gráfico 3: Mapa de calor com a correlação de Pearson entre as variáveis numéricas
# Objetivo: Identificar multicolinearidade e correlações lineares fortes entre as features
plt.figure(figsize=(10, 8))
matriz_correlacao = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(matriz_correlacao, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Fase 1 - Mapa de Calor (Correlação de Pearson)')
plt.savefig('heatmap_correlacao_pearson.png')
plt.close()

print("\nFase 1 concluída: Dimensões, métricas descritivas e gráficos analíticos gerados e salvos com sucesso.")
