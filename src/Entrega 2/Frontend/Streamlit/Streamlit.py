import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pytrends.request import TrendReq
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Dashboard Executivo de Vendas", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/2025-2-NCC5/Projeto2/main/cannoli_atualizado.csv')
        df['Data_Pedido'] = pd.to_datetime(df['Data_Pedido'], dayfirst=True, format='mixed')
        df['Dia_Semana'] = df['Data_Pedido'].dt.day_name()
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def get_trends_data():
    pytrends = TrendReq(hl='pt-BR', tz=360)
    keywords = ["pizza", "massas", "sobremesa"]
    try:
        # --- ALTERAÇÃO AQUI ---
        # Período de busca alterado DE VOLTA para 'today 3-m' (3 meses)
        pytrends.build_payload(keywords, cat=0, timeframe='today 3-m', geo='BR', gprop='')
        
        df_trends = pytrends.interest_over_time().reset_index()
        df_trends = df_trends.rename(columns={'date': 'Data_Pedido', 'pizza': 'trend_pizza', 'massas': 'trend_massas', 'sobremesa': 'trend_sobremesa'})
        df_trends['Data_Pedido'] = pd.to_datetime(df_trends['Data_Pedido']).dt.tz_localize(None)
        return df_trends[['Data_Pedido', 'trend_pizza', 'trend_massas', 'trend_sobremesa']]
    except Exception as e:
        st.error(f"Erro ao buscar dados do Google Trends: {e}. Tente novamente mais tarde.")
        return pd.DataFrame()

df_raw = load_data()
df_trends = get_trends_data()

if df_raw.empty:
    st.error("Erro ao carregar dados de vendas.")
    st.stop()

st.sidebar.header("Filtros Globais")
min_date = df_raw['Data_Pedido'].min().date()
max_date = df_raw['Data_Pedido'].max().date()
date_range = st.sidebar.date_input("Período de Análise", (min_date, max_date), min_value=min_date, max_value=max_date)

cats = ['Todas'] + list(df_raw['Categoria_Comida'].unique())
selected_cat = st.sidebar.selectbox("Categoria de Produto", cats)

if len(date_range) == 2:
    mask = (df_raw['Data_Pedido'].dt.date >= date_range[0]) & (df_raw['Data_Pedido'].dt.date <= date_range[1])
    df_filtered = df_raw.loc[mask]
else:
    df_filtered = df_raw

if selected_cat != 'Todas':
    df_filtered = df_filtered[df_filtered['Categoria_Comida'] == selected_cat]

df_daily = df_filtered.groupby('Data_Pedido').agg(
    Pedidos=('ID_Pedido', 'nunique'),
    Volume_Itens=('Quantidade', 'sum')
).reset_index()

if not df_trends.empty:
    df_daily = pd.merge(df_daily, df_trends, on='Data_Pedido', how='left').fillna(0)

st.title("📊 Dashboard Executivo de Vendas")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total de Pedidos", f"{df_filtered['ID_Pedido'].nunique():,.0f}")
with kpi2:
    st.metric("Volume de Itens Vendidos", f"{df_filtered['Quantidade'].sum():,.0f}")
with kpi3:
    media_itens = df_filtered['Quantidade'].sum() / df_filtered['ID_Pedido'].nunique() if df_filtered['ID_Pedido'].nunique() > 0 else 0
    st.metric("Média Itens/Pedido", f"{media_itens:.2f}")
with kpi4:
    st.metric("Média Pedidos/Dia", f"{df_daily['Pedidos'].mean():.1f}")
with kpi5:
    try:
        best_day = df_filtered.groupby('Dia_Semana')['Quantidade'].sum().idxmax()
        st.metric("Melhor Dia da Semana", best_day)
    except:
        st.metric("Melhor Dia da Semana", "-")


col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # --- ALTERAÇÃO AQUI ---
    # Substituído Média Móvel (que tinha um gap no início)
    # por uma Linha de Tendência (Loess) que cobre todo o período.
    st.subheader("Evolução Diária de Vendas (com Linha de Tendência)")
    
    base = alt.Chart(df_daily).encode(x=alt.X('Data_Pedido', axis=alt.Axis(title='Data', format='%d/%m')))
    
    bar = base.mark_bar(opacity=0.5, color='#1f77b4').encode(
        y=alt.Y('Volume_Itens', title='Volume Diário'),
        tooltip=['Data_Pedido', 'Volume_Itens']
    )
    
    # Linha de Tendência Suavizada (Loess)
    line = base.transform_loess(
        'Data_Pedido', 
        'Volume_Itens',
        bandwidth=0.3 # Ajuste (0.1 a 1.0) para mais ou menos suavidade
    ).mark_line(
        color='red', 
        size=3
    ).encode(
        y=alt.Y('Volume_Itens', title='Tendência')
    )
    
    st.altair_chart((bar + line).interactive(), use_container_width=True)

with col_main2:
    st.subheader("Mix de Categoria (Volume)")
    df_cat = df_filtered.groupby('Categoria_Comida')['Quantidade'].sum().reset_index()
    donut = alt.Chart(df_cat).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("Quantidade", stack=True),
        color=alt.Color("Categoria_Comida"),
        tooltip=["Categoria_Comida", "Quantidade"]
    )
    st.altair_chart(donut, use_container_width=True)

col_sec1, col_sec2 = st.columns(2)

with col_sec1:
    st.subheader("Performance por Dia da Semana")
    df_week = df_filtered.groupby('Dia_Semana')['Quantidade'].sum().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).reset_index()
    
    bar_week = alt.Chart(df_week).mark_bar().encode(
        x=alt.X('Dia_Semana', sort=None, title='Dia'),
        y=alt.Y('Quantidade', title='Volume Total'),
        color=alt.value('#2ca02c'),
        tooltip=['Dia_Semana', 'Quantidade']
    )
    st.altair_chart(bar_week, use_container_width=True)

with col_sec2:
    st.subheader("Análise de Correlação: Vendas vs. Google Trends")
    
    trend_map = {'Pizza': 'trend_pizza', 'Massas': 'trend_massas', 'Sobremesa': 'trend_sobremesa'}
    
    if selected_cat in trend_map and not df_trends.empty:
        trend_col = trend_map[selected_cat]
        
        df_corr = df_daily[['Volume_Itens', trend_col]].dropna()
        if len(df_corr) > 5 and df_corr[trend_col].nunique() > 1:
            X = df_corr[[trend_col]]
            y = df_corr['Volume_Itens']
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            
            scatter = alt.Chart(df_corr).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X(trend_col, title=f"Índice Google Trends ({selected_cat})", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Volume_Itens', title='Volume de Vendas'),
                tooltip=['Data_Pedido', trend_col, 'Volume_Itens']
            )
            
            reg_line = scatter.transform_regression(
                trend_col, 'Volume_Itens',
                extent=[0, 100] 
            ).mark_line(
                color='red',
                strokeWidth=3 
            )
            
            st.altair_chart(scatter + reg_line, use_container_width=True)
            st.caption(f"Coeficiente de Determinação (R²): {r2:.4f}")
        else:
            st.warning("Dados insuficientes ou constantes para correlação.")
    else:
        st.info("Selecione uma categoria específica (Pizza, Massas, Sobremesa) na barra lateral para ver a correlação com o Google Trends.")

st.subheader("Detalhamento dos Top Produtos (Curva ABC)")
df_items = df_filtered.groupby('Nome_do_Item').agg(
    Pedidos=('ID_Pedido', 'nunique'),
    Total_Itens=('Quantidade', 'sum')
).sort_values('Total_Itens', ascending=False)

if not df_items.empty:
    df_items['% do Total'] = (df_items['Total_Itens'] / df_items['Total_Itens'].sum()) * 100
    df_items['% Acumulado'] = df_items['% do Total'].cumsum()

    st.dataframe(
        df_items.style.format({'% do Total': '{:.1f}%', '% Acumulado': '{:.1f}%'}),
        use_container_width=True
    )
else:
    st.info("Nenhum item encontrado para os filtros selecionados.")


# --- SIMULADOR BASEADO NA LINHA VERMELHA ---
sim_model_exists = 'model' in locals() and 'r2' in locals() and r2 > 0 and selected_cat in trend_map

if sim_model_exists:
    st.divider()
    st.subheader(f"🔮 Simulador de Vendas para {selected_cat}")
    
    trend_col = trend_map[selected_cat]
    
    trend_simulado = st.slider(
        f"Selecione um valor de Busca Google ({selected_cat})", 
        min_value=0, 
        max_value=100, 
        value=int(df_daily[trend_col].max()) if df_daily[trend_col].max() > 0 else 50
    )
    
    venda_prevista = model.predict([[trend_simulado]])[0]
    
    st.metric(
        label=f"Vendas Previstas de {selected_cat}",
        value=f"{venda_prevista:.0f} itens",
        delta=f"{venda_prevista - df_filtered[df_filtered['Categoria_Comida'] == selected_cat]['Quantidade'].mean():.0f} vs Média Atual" if not df_filtered[df_filtered['Categoria_Comida'] == selected_cat].empty else None
    )
    st.caption("Esta simulação usa a linha de regressão vermelha (acima) para prever as vendas.")