# 🚨 Detecção de Anomalias em Transações Financeiras

Este projeto foi desenvolvido como parte de um curso da **DIO (Digital Innovation One)**, com o objetivo de aplicar técnicas de **Machine Learning** para identificar possíveis fraudes em transações financeiras. 💳🤖

A proposta do projeto é construir um pipeline de análise e modelagem utilizando Python, desde o carregamento dos dados até a avaliação de modelos de classificação.

## 📌 Sobre o Projeto

O dataset utilizado contém transações realizadas com cartão de crédito, onde a coluna `Class` indica se a transação é normal ou fraudulenta:

* `0` → Transação normal ✅
* `1` → Transação fraudulenta ⚠️

Como esse tipo de problema possui uma grande diferença entre a quantidade de transações normais e fraudulentas, o projeto também aplica técnicas para lidar com **dados desbalanceados**, como o **undersampling**.

## 🧠 Técnicas Utilizadas

Durante o desenvolvimento do projeto, foram aplicadas as seguintes etapas:

* 📥 Carregamento dos dados a partir de um arquivo CSV
* 📊 Análise da distribuição das classes
* 🔄 Transformação logarítmica na coluna `Amount`
* ⚖️ Padronização de variáveis numéricas
* ✂️ Separação dos dados em treino e teste
* 📉 Aplicação de undersampling para balanceamento dos dados
* 🤖 Treinamento com Regressão Logística
* 🌲 Treinamento com Random Forest
* 📈 Avaliação com métricas de classificação
* 📌 Geração de curva ROC
* 📌 Geração de curva Precision-Recall
* 📝 Salvamento dos resultados em arquivo `.txt`

## 🛠️ Tecnologias Utilizadas

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Imbalanced-learn

## 📊 Modelos Utilizados

Foram testados dois modelos de classificação:

### 🔹 Regressão Logística

Modelo simples e eficiente para problemas de classificação binária, utilizado como baseline do projeto.

### 🔹 Random Forest

Modelo baseado em árvores de decisão, utilizado para comparar o desempenho com a Regressão Logística.

## 📁 Resultados

O projeto gera um arquivo `results.txt` contendo os principais resultados da execução do pipeline, incluindo:

* Distribuição original das classes
* Distribuição após undersampling
* Relatório de classificação da Regressão Logística
* Relatório de classificação do Random Forest
* Valores de AUC
* Referência aos gráficos gerados

Também são salvos gráficos com a distribuição das classes antes e depois do balanceamento.

## 🎯 Objetivo de Aprendizado

Este projeto teve como objetivo praticar conceitos importantes de Ciência de Dados e Machine Learning, como:

* Pré-processamento de dados
* Tratamento de dados desbalanceados
* Classificação binária
* Avaliação de modelos
* Organização de código em funções
* Criação de pipeline de Machine Learning

## 🚀 Status do Projeto

✅ Projeto em desenvolvimento como parte dos estudos na DIO.


