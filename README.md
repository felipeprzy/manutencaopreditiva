# Projeto Avaliativo Módulo 1: Pipeline Preditivo para Manutenção na Indústria 4.0

## 📝 Descrição do Projeto
Repositório oficial desenvolvido para a implementação de um pipeline de **Manutenção Preditiva** baseado em Machine Learning, com foco na previsão de quebras mecânicas em equipamentos industriais, prevenção de *Data Leakage*, tratamento estatístico de dados e interpretabilidade de modelos classificadores.

---

## 📂 Estrutura do Repositório
O projeto está centralizado no Notebook principal **`projeto_avaliativo_modulo_1.ipynb`**, que executa de forma estruturada as seguintes etapas:

- **Fase 1 (EDA):** Análise Exploratória de Dados, verificação de dimensões, tipos de dados, resumo estatístico descritivo e geração dos gráficos analíticos (Histograma, Desbalanceamento do Target e Correlação de Pearson).
- **Fase 2 (Data Prep):** Limpeza de registros duplicados, tratamento de valores ausentes (NaN) com justificativa técnica baseada em média/mediana, e geração de diagramas Boxplot para análise de outliers.
- **Fase 3 (Feature Engineering):** Criação de nova variável numérica preditiva (`potencia` através da multiplicação de rotação por torque) para potencializar a detecção de anomalias operacionais.
- **Fase 4 (Divisão e Balanceamento):** Separação estratificada (Train-Test Split 80/20 com `stratify=y`) e aplicação da técnica SMOTE **exclusivamente no conjunto de treino** para evitar *Data Leakage*.
- **Fase 5 (Escalonamento):** Aplicação de `StandardScaler` apenas nos dados destinados ao modelo KNN (`fit_transform` no treino e `transform` no teste), mantendo a Árvore de Decisão sem escalonamento devido à sua imunidade a escalas.
- **Fase 6 (Ajuste de Parâmetros e Overfitting):** Treinamento e variação de hiperparâmetros ($K \in \{3, 5, 7\}$ para o KNN e `max_depth` $\in \{3, 5, \text{None}\}$ para a Árvore), registrando e avaliando a estabilidade entre treino e teste.
- **Fase 7 (Veredito Final):** Cálculo da acurácia final em conjunto de teste independente e emissão da conclusão técnica comparativa indicando o modelo ideal para adoção fabril.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
O projeto foi desenvolvido em **Python** utilizando o seguinte ecossistema de Data Science:

- `pandas` e `numpy` (Manipulação e computação numérica)
- `matplotlib` e `seaborn` (Visualização de dados e gráficos analíticos)
- `scikit-learn` (Pré-processamento, StandardScaler, KNN, Árvore de Decisão e métricas de acurácia)
- `imbalanced-learn` (Técnica de reamostragem SMOTE para dados desbalanceados)

---

## 🚀 Instruções de Execução

1. **Baixe o Notebook:** Obtenha o arquivo `projeto_avaliativo_modulo_1.ipynb`.
   https://colab.research.google.com/drive/1en5nR0K_E0hKttjsFSMxzrGXaqKlXWOv
3. **Ambiente de Execução:** Faça o upload do notebook e da base de dados **`manutencao_preditiva.csv`** para o ambiente do **Google Colab**.
4. **Execução:** Execute todas as células em ordem sequencial para rodar o pipeline ponta a ponta.

---

## 🎥 Vídeo de Apresentação
- **Link do Vídeo:** [Inserir link do Google Drive em modo leitor para qualquer pessoa com o link]
