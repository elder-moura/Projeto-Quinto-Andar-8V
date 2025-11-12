import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from PIL import Image # Usado para carregar as imagens

# Ignorar avisos
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# FUNÇÃO PARA CARREGAR OS ARQUIVOS
# ---------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo treinado."""
    try:
        modelo = joblib.load('modelo_aluguel_rf.pkl')
        return modelo
    except FileNotFoundError:
        st.error("Arquivo do modelo 'modelo_aluguel_rf.pkl' não encontrado.")
        st.info("Certifique-se de que o arquivo .pkl está no mesmo diretório do app.py no GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}. Verifique a versão do scikit-learn no requirements.txt.")
        st.stop()

@st.cache_data
def carregar_bairros():
    """Carrega a lista de bairros únicos."""
    try:
        with open('bairros_unicos.json', 'r', encoding='utf-8') as f:
            bairros = json.load(f)
        return bairros
    except FileNotFoundError:
        st.error("Arquivo de bairros 'bairros_unicos.json' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar a lista de bairros: {e}")
        return []

# --- Carregar os dados ---
modelo_pipeline = carregar_modelo()
bairros_unicos = carregar_bairros()
st.set_page_config(layout="wide")


# ---------------------------------------------------------------------
# INTERFACE DO USUÁRIO (Inputs na Barra Lateral)
# ---------------------------------------------------------------------

st.sidebar.title("🏙️ Estimador de Aluguel")
st.sidebar.markdown("Preencha os dados do imóvel para fazer uma previsão.")

# Inputs na barra lateral
st.sidebar.header("Características do Imóvel")

metragem = st.sidebar.number_input("Metragem (m²)", min_value=10, max_value=1000, value=70, step=5)
quartos = st.sidebar.selectbox("Quartos", options=[0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
banheiros = st.sidebar.selectbox("Banheiros", options=[0, 1, 2, 3, 4, 5, 6], index=1)
vagas = st.sidebar.selectbox("Vagas de Garagem", options=[0, 1, 2, 3, 4, 5], index=1)

bairro_default_index = 0
if 'aclimacao' in bairros_unicos:
    bairro_default_index = bairros_unicos.index('aclimacao')
bairro = st.sidebar.selectbox("Bairro", options=bairros_unicos, index=bairro_default_index)

# Botão para prever
prever = st.sidebar.button("Estimar Valor", type="primary")

# ---------------------------------------------------------------------
# LAYOUT PRINCIPAL COM ABAS
# ---------------------------------------------------------------------

st.title("Simulador e Análise de Mercado de Aluguéis")
tab1, tab2 = st.tabs(["🏠 Simulador", "📊 Análise de Mercado"])

# --- ABA 1: SIMULADOR ---
with tab1:
    st.header("Resultado da Simulação")
    
    if prever:
        try:
            # 1. Criar DataFrame de entrada
            input_data = pd.DataFrame({
                'Metragem': [metragem], 'Quartos': [quartos], 'Banheiros': [banheiros],
                'Vagas': [vagas],
                'Bairro': [bairro]
            })
            
            # 2. Fazer a previsão
            previsao = modelo_pipeline.predict(input_data)[0]
            preco_formatado = f"R$ {previsao:,.2f}"
            
            st.success(f"## Valor Total Estimado: {preco_formatado}")
            
            st.markdown("---")
            st.subheader("Resumo dos Dados Informados:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Metragem:** {metragem} m²")
                st.write(f"**Quartos:** {quartos}")
                st.write(f"**Banheiros:** {banheiros}")
                st.write(f"**Vagas:** {vagas}")
            with col2:
                st.write(f"**Bairro:** {bairro.title()}")
        
        except Exception as e:
            st.error(f"Erro ao realizar a previsão: {e}")
    
    else:
        st.info("Preencha os dados na barra lateral e clique em 'Estimar Valor'.")

    st.markdown("---")
    st.info(
        "**Sobre o Modelo:**\n"
        f"* **Modelo Utilizado:** Random Forest Regressor (R²: 0.901)\n"
        f"* **Base de Dados:** 11.283 imóveis (após limpeza)"
    )

# --- ABA 2: ANÁLISE DE MERCADO ---
with tab2:
    st.header("Análise Exploratória dos Dados")
    st.write("Estes gráficos são baseados nos 11.283 imóveis da base de dados e foram gerados durante o treino do modelo.")

    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image('graf_top_bairros_caros.png', caption='Top 10 Bairros por Preço Médio Total')
            st.image('graf_metragem_vs_preco.png', caption='Metragem vs. Preço Total')
            
        with col2:
            st.image('graf_top_bairros_valorizados.png', caption='Top 15 Bairros por Preço Médio/m²')
            st.image('graf_quartos_vs_preco.png', caption='Preço por Número de Quartos')
            
    except Exception as e:
        st.error(f"Erro ao carregar gráficos: {e}")

        st.info("Certifique-se de que os arquivos 'graf_*.png' estão no repositório do GitHub.")
