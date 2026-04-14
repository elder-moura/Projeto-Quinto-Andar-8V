import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

# Ignorar avisos
warnings.filterwarnings("ignore")

# Configuração da página para ocupar mais espaço e ter um título na aba do navegador
st.set_page_config(page_title="Estimador de Aluguel", page_icon="🏙️", layout="wide")

# ---------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E LÓGICA (BACKEND DO STREAMLIT)
# ---------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo treinado."""
    try:
        return joblib.load('modelo_aluguel_5vars.pkl') 
    except FileNotFoundError:
        st.error("🚨 Arquivo 'modelo_aluguel_5vars.pkl' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Erro ao carregar o modelo. Verifique a versão do scikit-learn. Detalhe: {e}")
        st.stop()

@st.cache_data
def carregar_bairros():
    """Carrega a lista de bairros únicos."""
    try:
        with open('bairros_unicos.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return ["Bairro Padrão"] # Fallback caso o arquivo falhe

def realizar_previsao(modelo, metragem, quartos, banheiros, vagas, bairro):
    """Encapsula a lógica de montagem do DataFrame e inferência."""
    input_data = pd.DataFrame({
        'Metragem': [metragem],
        'Quartos': [quartos],
        'Banheiros': [banheiros],
        'Vagas': [vagas],
        'Bairro': [bairro]
    })
    return modelo.predict(input_data)[0]

# Carregar os artefatos
modelo_pipeline = carregar_modelo()
bairros_unicos = carregar_bairros()

# ---------------------------------------------------------------------
# INTERFACE DO USUÁRIO (FRONTEND DO STREAMLIT)
# ---------------------------------------------------------------------

st.title("🏙️ Estimador de Aluguel Inteligente")
st.markdown("Descubra o valor ideal do aluguel com base nas características do imóvel.")
st.divider()

# Layout em colunas para a área principal
col_inputs, col_resultados = st.columns([1, 2], gap="large")

with col_inputs:
    st.subheader("⚙️ Características")
    
    # Inputs organizados
    bairro = st.selectbox("📍 Bairro", options=bairros_unicos)
    metragem = st.number_input("📏 Metragem (m²)", min_value=20, max_value=500, value=65, step=1)
    
    # Usando colunas internas para inputs menores
    c1, c2 = st.columns(2)
    with c1:
        quartos = st.selectbox("🛏️ Quartos", options=[0, 1, 2, 3, 4, 5, 6], index=2)
        vagas = st.selectbox("🚗 Vagas", options=[0, 1, 2, 3, 4, 5], index=1)
    with c2:
        banheiros = st.selectbox("🚿 Banheiros", options=[0, 1, 2, 3, 4, 5], index=1)
    
    # Botão de ação destacado
    calcular = st.button("📊 Estimar Valor", type="primary", use_container_width=True)

with col_resultados:
    if calcular:
        try:
            with st.spinner("Analisando o mercado..."):
                previsao = realizar_previsao(modelo_pipeline, metragem, quartos, banheiros, vagas, bairro)
                preco_m2 = previsao / metragem if metragem > 0 else 0
            
            st.subheader("Resultado da Avaliação")
            
            # Data Storytelling: Exibindo os dados em formato de "Cards de KPI"
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(label="Valor Total Estimado (Mensal)", value=f"R$ {previsao:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                
            with metric_col2:
                st.metric(label="Preço Médio por m²", value=f"R$ {preco_m2:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            
            st.success(f"O modelo Random Forest encontrou o valor de **R$ {previsao:,.2f}** considerando o padrão histórico para imóveis de {metragem}m² no bairro **{bairro.title()}**.")
            
        except Exception as e:
            st.error(f"Erro ao realizar a previsão: {e}")
    else:
        st.info("👈 Preencha as características do imóvel ao lado e clique em 'Estimar Valor' para ver a análise.")
