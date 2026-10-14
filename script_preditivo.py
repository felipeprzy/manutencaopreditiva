import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Carregamento dos dados
df = pd.read_csv('manutencao_preditiva.csv')

print(f"Dimensões do Dataset: {df.shape[0]} linhas e {df.shape[1]} colunas\n")
print("=== Tipos de Dados ===")
print(df.dtypes)
print("\n=== Resumo Estatístico Descritivo (.describe()) ===")
display(df.describe())

# 1. Histograma das variáveis preditoras
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'falha_maquina' in num_cols:
    num_cols.remove('falha_maquina')

df[num_cols].hist(bins=20, figsize=(12, 8), color='steelblue', edgecolor='black')
plt.suptitle("Histograma de Distribuição das Variáveis Preditoras", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Gráfico de barras: Desbalanceamento do Target (falha_maquina)
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='falha_maquina', data=df, palette='Set2')
plt.title("Taxa de Desbalanceamento da Variável Alvo (falha_maquina)", fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()

# 3. Mapa de Calor (Correlação de Pearson)
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Calor - Correlação de Pearson", fontsize=14, fontweight='bold')
plt.show()

# Remoção de duplicadas
duplicadas = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Linhas duplicadas removidas: {duplicadas}")

# Imputação de nulos por média ou mediana com base na assimetria
for col in num_cols:
    if df[col].isnull().sum() > 0:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            val = df[col].median()
            df[col].fillna(val, inplace=True)
            print(f"Coluna '{col}': Imputada por MEDIANA (Assimetria: {skew_val:.2f})")
        else:
            val = df[col].mean()
            df[col].fillna(val, inplace=True)
            print(f"Coluna '{col}': Imputada por MÉDIA (Assimetria: {skew_val:.2f})")

# Boxplot para outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[num_cols], palette="pastel")
plt.title("Análise de Outliers - Boxplot", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
