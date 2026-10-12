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


# ==============================================================================
# FASE 2: LIMPEZA E TRATAMENTO DE DADOS (DATA PREP)
# ==============================================================================

print("\n--- INICIANDO FASE 2: LIMPEZA E TRATAMENTO DE DADOS ---")

# ------------------------------------------------------------------------------
# 1. Tratamento de Registros Duplicados
# ------------------------------------------------------------------------------
# Justificativa técnica: Registros duplicados inserem viés de sobre-representação 
# no modelo preditivo, inflando artificialmente a acurácia. A remoção preserva 
# a independência estatística das observações.
total_duplicados_antes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Registros duplicados identificados e removidos: {total_duplicados_antes}")

# ------------------------------------------------------------------------------
# 2. Tratamento de Valores Ausentes (Missing Values / NaN)
# ------------------------------------------------------------------------------
# Justificativa técnica: Modelos de Machine Learning (como KNN e Árvores de Decisão) 
# não processam valores nulos nativamente. Optou-se pela substituição utilizando a 
# MEDIANA para colunas numéricas, pois, diferentemente da média, a mediana é uma 
# medida robusta que não sofre distorções severas na presença de valores discrepantes (outliers).
colunas_com_nulos = df.columns[df.isnull().any()]

if len(colunas_com_nulos) > 0:
    for coluna in colunas_com_nulos:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            mediana_calculada = df[coluna].median()
            df[coluna] = df[coluna].fillna(mediana_calculada)
            print(f"Coluna numérica '{coluna}': valores nulos preenchidos com a mediana ({mediana_calculada})")
        else:
            moda_calculada = df[coluna].mode()[0]
            df[coluna] = df[coluna].fillna(moda_calculada)
            print(f"Coluna categórica '{coluna}': valores nulos preenchidos com a moda ({moda_calculada})")
else:
    print("Nenhum valor ausente (NaN) detectado no dataset.")

# ------------------------------------------------------------------------------
# 3. Análise e Visualização de Outliers (Boxplot)
# ------------------------------------------------------------------------------
# Justificativa técnica: Valores extremos (outliers) podem desestabilizar o cálculo 
# de distâncias euclidianas no algoritmo KNN. Geramos o gráfico Boxplot para 
# inspeção visual da dispersão e identificação de assimetrias severas.
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]), palette="Set3")
plt.title('Fase 2 - Identificação de Outliers nas Variáveis Numéricas (Boxplot)')
plt.xlabel('Variáveis do Processo')
plt.ylabel('Valores / Escala')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_analise_outliers.png')
plt.close()

print("\nFase 2 concluída com sucesso: Dados limpos, duplicatas removidas, nulos tratados por robustez estatística e boxplots gerados.")


# ==============================================================================
# FASE 3: FEATURE ENGINEERING (ENGENHARIA DE RECURSOS)
# ==============================================================================

print("\n--- INICIANDO FASE 3: ENGENHARIA DE RECURSOS ---")

# Justificativa técnica: A engenharia de recursos consiste na transformação e criação 
# de novas variáveis preditoras a partir dos dados brutos existentes. O objetivo é 
# fornecer interações físicas e estatísticas mais evidentes para os algoritmos de 
# Machine Learning, potencializando a capacidade de detecção precoce de falhas.

# Seleção automática das colunas numéricas disponíveis no dataset limpo
colunas_numericas = df.select_dtypes(include=[np.number]).columns

if len(colunas_numericas) >= 2:
    # ------------------------------------------------------------------------------
    # 1. Criação de Feature por Interação Multiplicativa (Proxy de Esforço Operacional)
    # ------------------------------------------------------------------------------
    # Lógica matemática: Multiplicar duas variáveis de sensores interdependentes 
    # (ex: torque e rotação, ou pressão e temperatura) gera um indicador composto 
    # que reflete o estresse mecânico ou térmico acumulado no componente industrial.
    col1, col2 = colunas_numericas[0], colunas_numericas[1]
    df['feature_esforco_combinado'] = df[col1] * df[col2]
    print(f"Feature criada com sucesso: 'feature_esforco_combinado' (Produto entre '{col1}' e '{col2}').")

if len(colunas_numericas) >= 3:
    # ------------------------------------------------------------------------------
    # 2. Criação de Feature por Razão / Quociente (Taxa de Eficiência Operacional)
    # ------------------------------------------------------------------------------
    # Lógica matemática: A divisão entre variáveis correlacionadas normaliza a 
    # escala de operação. Adiciona-se uma constante infinitesimal (1e-5) no denominador 
    # para prevenir erros de divisão por zero (ArithmeticError) em leituras nulas de sensores.
    col3 = colunas_numericas[2]
    df['feature_taxa_operacional'] = df[col1] / (df[col3].abs() + 1e-5)
    print(f"Feature criada com sucesso: 'feature_taxa_operacional' (Razão entre '{col1}' e '{col3}').")

print("\nFase 3 concluída com sucesso: Variáveis sintéticas adicionadas ao pipeline preditivo.")

# ==============================================================================
# FASE 4: DIVISÃO DO DATASET, ESTRATIFICAÇÃO, BALANCEAMENTO E O USO DO .fit()
# ==============================================================================

print("\n--- INICIANDO FASE 4: PREPARAÇÃO DOS DADOS PARA MODELAGEM ---")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------------
# 1. Separação entre Variáveis Explicativas (X) e Variável Alvo (y)
# ------------------------------------------------------------------------------
# Assume-se que a última coluna do DataFrame seja a variável alvo (rótulo de falha)
# e todas as demais colunas anteriores sejam as features preditoras dos sensores.
coluna_alvo = df.columns[-1]
X = df.drop(columns=[coluna_alvo])
y = df[coluna_alvo]

print(f"Variável Alvo definida: '{coluna_alvo}'")
print(f"Total de features preditoras (X): {X.shape[1]}")

# ------------------------------------------------------------------------------
# 2. Divisão Estratificada em Conjuntos de Treino e Teste (Train-Test Split)
# ------------------------------------------------------------------------------
# Justificativa técnica: A estratificação (stratify=y) garante que a proporção 
# de classes (normal vs falha) seja idêntica tanto no conjunto de treino quanto no 
# de teste, prevenindo desvios em cenários de dados desbalanceados.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

print(f"Dimensões do conjunto de treino (X_train): {X_train.shape}")
print(f"Dimensões do conjunto de teste (X_test): {X_test.shape}")

# ------------------------------------------------------------------------------
# 3. APLICAÇÃO RIGOROSA DO MÉTODO .fit() E PREVENÇÃO DE DATA LEAKAGE
# ------------------------------------------------------------------------------
# EXPLICAÇÃO TÉCNICA OBRIGATÓRIA SOBRE O .fit():
# O método .fit() calcula os parâmetros estatísticos do transformador ou modelo 
# (como média e desvio padrão para padronização, ou regras de fronteira).
# 
# REGRA DE OURO CONTRA DATA LEAKAGE (Vazamento de Dados):
# - O .fit() DEVE ser executado EXCLUSIVAMENTE sobre o conjunto de treino (X_train).
# - No conjunto de teste (X_test), aplicamos estritamente o .transform().
# 
# Por que isso é vital? Se aplicarmos o .fit() no dataset inteiro antes da divisão, 
# o modelo absorve estatísticas globais (incluindo o teste), gerando métricas de 
# desempenho falsamente infladas que falharão em ambiente de produção industrial.

# Instanciação do escalonador de dados
scaler = StandardScaler()

# Aplicação do .fit_transform() SOMENTE no conjunto de treino
X_train_scaled = scaler.fit_transform(X_train)

# Aplicação APENAS do .transform() no conjunto de teste (respeitando a blindagem)
X_test_scaled = scaler.transform(X_test)

print("Escalonamento (StandardScaler) aplicado com sucesso, respeitando a blindagem do .fit() no treino.")

# ------------------------------------------------------------------------------
# 4. Tratamento de Desbalanceamento de Classes com SMOTE (Synthetic Minority Over-sampling)
# ------------------------------------------------------------------------------
# Justificativa técnica: Em manutenção preditiva, falhas reais são eventos raros, 
# gerando forte desbalanceamento no dataset. O SMOTE gera amostras sintéticas 
# da classe minoritária estritamente no conjunto de treino (X_train_scaled) para 
# evitar o vazamento de informações para os dados de teste.
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Balanceamento SMOTE concluído. Novas dimensões de treino balanceado: {X_train_balanced.shape}")


