# Pipeline Preditivo Inteligente - Indústria 4.0

Repositório oficial desenvolvido para a implementação de um pipeline de **Manutenção Preditiva** baseado em Machine Learning, seguindo rigorosamente padrões acadêmicos e industriais de Ciência de Dados (prevenção de *Data Leakage*, validação cruzada e interpretabilidade de modelos).

---

## 📂 Estrutura do Repositório

O projeto está centralizado no script principal `script_preditivo.py`, que executa de forma modular as seguintes etapas:

* **Fase 1 (EDA):** Análise Exploratória de Dados, verificação de dimensões, tipos de dados, resumo estatístico e geração dos gráficos analíticos (Histograma, Desbalanceamento e Correlação de Pearson).
* **Fase 2 (Data Prep):** Limpeza de registros duplicados, tratamento de valores ausentes (NaN) utilizando a mediana para robustez contra outliers, e geração de diagramas Boxplot.
* **Fase 3 (Feature Engineering):** Criação de novas variáveis preditivas (interações e razões matemáticas) para potencializar a detecção de anomalias operacionais.
* **Fase 4 (Divisão e Blindagem):** Separação estratificada (Train-Test Split), escalonamento com `StandardScaler` (aplicando o `.fit()` **estritamente** no conjunto de treino para evitar *Data Leakage*) e balanceamento de classes via **SMOTE**.
* **Fase 5 (Modelagem):** Treinamento, otimização de hiperparâmetros via `GridSearchCV` e validação cruzada para os algoritmos **K-Nearest Neighbors (KNN)** e **Árvore de Decisão**.
* **Fase 6 (Avaliação):** Emissão de relatórios de classificação, cálculo de métricas ROC-AUC e geração das Matrizes de Confusão e Curvas ROC comparativas.
* **Fase 7 (Consolidação):** Exportação dos resultados preditivos consolidados para arquivos de auditoria em formato CSV.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O script foi desenvolvido em **Python** utilizando o seguinte ecossistema de Data Science:
* `pandas` e `numpy` (Manipulação e computação numérica)
* `matplotlib` e `seaborn` (Visualização de dados e gráficos analíticos)
* `scikit-learn` (Pré-processamento, métricas, KNN, Árvore de Decisão e GridSearchCV)
* `imblearn` (Técnica de reamostragem SMOTE para dados desbalanceados)

---

## 🚀 Instruções de Execução

1. Certifique-se de que o dataset institucional em formato `.csv` está devidamente posicionado no diretório do projeto.
2. Atualize o caminho de leitura do arquivo no início do script `script_preditivo.py`, caso necessário.
3. Execute o script completo via terminal:
   ```bash
   python script_preditivo.py
