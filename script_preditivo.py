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
