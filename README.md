# 🏢 SextoAndar - Estimador Inteligente de Aluguel Residencial em São Paulo

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)

> **Trabalho de Conclusão de Curso (TCC) - Faculdade Impacta**  
> *Aplicação de Aprendizado de Máquina Supervisionado para Precificação Preditiva de Imóveis Residenciais na Cidade de São Paulo.*

---

## 📌 Visão Geral do Projeto

A precificação de aluguéis residenciais costuma ser um processo subjetivo ou baseado em buscas manuais assistemáticas. O **SextoAndar** é uma solução *data-driven* que estima em tempo real o valor de locação de imóveis com base em características como metragem, número de quartos, banheiros e localização (bairro).

O projeto cobriu o ciclo completo de vida dos dados, seguindo a metodologia **CRISP-DM**:
1. **Coleta de Dados:** Web Scraping automatizado com Selenium (11.283 registros limpos de SP).
2. **Tratamento & Limpeza:** Normalização de texto, remoção de outliers via quantis (P99) e engenharia de atributos.
3. **Modelagem Preditiva:** Comparação e validação de 4 algoritmos com **K-Fold Cross Validation**.
4. **Implantação (Deploy):** Interface interativa desenvolvida com **Streamlit**.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python
* **Coleta & Manipulação de Dados:** Selenium, Pandas, NumPy, Regex, Unidecode
* **Machine Learning & Pré-processamento:** Scikit-learn (`ColumnTransformer`, `OneHotEncoder`, `KFold`), XGBoost
* **Visualização de Dados:** Matplotlib, Seaborn
* **Interface & Deploy:** Streamlit

---

## 📊 Metodologia e Pipeline de Dados

### 1. Web Scraping & Resiliência
Devido a limitações em datasets públicos (dados inconsistentes/desatualizados), desenvolveu-se uma esteira própria de extração via **Selenium**:
* Manipulação de renderização dinâmica (JavaScript) e *scroll* infinito.
* Sistema de pausa aleatória (*delays*) para mitigação de IP Banning.
* Arquitetura em lotes (*batch processing*) para persistência em disco contra quedas de sessão.
* Controle de unicidade de dados por URL/ID para evitar duplicações.

### 2. Modelagem & Validação Cruzada
Foram avaliados 4 algoritmos representativos de diferentes paradigmas, utilizando **Validação Cruzada K-Fold (k=5)** para evitar *overfitting*:

| Modelo | MAE (R$) | RMSE (R$) | R² | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **R$ 269,63** | **R$ 581,79** | **0.877** | 🏆 **Selecionado** |
| XGBoost | R$ 549,48 | R$ 799,80 | 0.768 | Avaliado |
| Regressão Linear | R$ 569,99 | R$ 878,07 | 0.721 | Baseline |
| MLP Regressor (Rede Neural) | R$ 573,86 | R$ 888,59 | 0.714 | Avaliado |

> **Resultado:** O modelo **Random Forest** apresentou a maior precisão preditiva ($R^2 = 0.877$), sendo o escolhido para o motor do aplicativo.

---
