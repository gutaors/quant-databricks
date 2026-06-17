# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎯 Simulador de Investimentos
# MAGIC Simule compra e venda de ações com dados históricos.
# MAGIC
# MAGIC Equivalente a `pages/simulador.py` do Streamlit.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import plotly.graph_objs as go
from datetime import date, datetime, timedelta

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown("ticker", "", listar_tickers_disponiveis(spark), "Selecione o ativo")
dbutils.widgets.text("data_compra", "2024-01-02", "Data de Compra (AAAA-MM-DD)")
dbutils.widgets.text("data_venda", date.today().strftime("%Y-%m-%d"), "Data de Venda (AAAA-MM-DD)")
dbutils.widgets.text("valor_investimento", "1000.00", "Valor de investimento (R$)")

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
data_compra_sim = pd.to_datetime(dbutils.widgets.get("data_compra")).date()
data_venda_sim = pd.to_datetime(dbutils.widgets.get("data_venda")).date()
valor_sim = float(dbutils.widgets.get("valor_investimento"))

df_valores = carregar_cotacoes_ticker(spark, ticker)

if df_valores.empty:
    displayHTML(html_alert(f"Nenhum dado encontrado para <b>{ticker}</b>. Use o notebook 03_importa_cotacoes para importar.", "error"))
    dbutils.notebook.exit("Sem dados")

displayHTML(f"<h2>Simulação — {ticker}</h2>")
print(f"Período disponível: {df_valores['Date'].min().date()} a {df_valores['Date'].max().date()}")
print(f"Total de registros: {len(df_valores)}")

# COMMAND ----------

# ── Simulação de Compra ───────────────────────────────────────────────────────

# Encontrar dados na data de compra (ou próxima disponível)
dados_compra = df_valores[df_valores["Date"].dt.date == data_compra_sim]
if dados_compra.empty:
    proxima = df_valores[df_valores["Date"].dt.date > data_compra_sim]["Date"].min()
    if pd.isna(proxima):
        displayHTML(html_alert(f"Não há dados após {formatar_data(data_compra_sim)}", "error"))
        dbutils.notebook.exit("Sem dados para compra")
    dados_compra = df_valores[df_valores["Date"] == proxima]
    displayHTML(html_alert(f"Usando próxima data disponível para compra: {formatar_data(proxima)}", "warning"))

close_compra = float(dados_compra.iloc[0]["Close"])
quantidade_comprada = valor_sim / close_compra
data_compra_real = dados_compra.iloc[0]["Date"].date()

# COMMAND ----------

# ── Simulação de Venda ────────────────────────────────────────────────────────

dados_venda = df_valores[df_valores["Date"].dt.date == data_venda_sim]
if dados_venda.empty:
    anterior = df_valores[df_valores["Date"].dt.date < data_venda_sim]["Date"].max()
    if pd.isna(anterior):
        displayHTML(html_alert(f"Não há dados antes de {formatar_data(data_venda_sim)}", "error"))
        dbutils.notebook.exit("Sem dados para venda")
    dados_venda = df_valores[df_valores["Date"] == anterior]
    displayHTML(html_alert(f"Usando data anterior disponível para venda: {formatar_data(anterior)}", "warning"))

close_venda = float(dados_venda.iloc[0]["Close"])
data_venda_real = dados_venda.iloc[0]["Date"].date()
valor_vendido = quantidade_comprada * close_venda
lucro_prejuizo = valor_vendido - valor_sim
pct = (lucro_prejuizo / valor_sim) * 100

# COMMAND ----------

# ── Resultados ────────────────────────────────────────────────────────────────

# Valores extremos no período
df_periodo = df_valores[
    (df_valores["Date"].dt.date >= data_compra_real) &
    (df_valores["Date"].dt.date <= data_venda_real)
]

cor = "#0f9d58" if lucro_prejuizo >= 0 else "#dc3545"

metrics_html = f"""
<div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
    {html_metric("Quantidade", f"{quantidade_comprada:.2f}")}
    {html_metric("Preço Compra", formatar_moeda(close_compra))}
    {html_metric("Preço Venda", formatar_moeda(close_venda))}
"""

if not df_periodo.empty:
    max_periodo = float(df_periodo["Close"].max())
    min_periodo = float(df_periodo["Close"].min())
    data_max = df_periodo[df_periodo["Close"] == df_periodo["Close"].max()]["Date"].iloc[0]
    data_min = df_periodo[df_periodo["Close"] == df_periodo["Close"].min()]["Date"].iloc[0]
    metrics_html += f"""
    {html_metric("Maior no Período", formatar_moeda(max_periodo), f"Em {formatar_data(data_max)}")}
    {html_metric("Menor no Período", formatar_moeda(min_periodo), f"Em {formatar_data(data_min)}")}
    """

metrics_html += f"""
</div>
<div style='background:{cor}; color:white; padding:20px; border-radius:10px; text-align:center; margin:16px 0;'>
    <h2 style='margin:0; color:white;'>Resultado: {formatar_moeda(lucro_prejuizo)} ({pct:.2f}%)</h2>
    <p style='margin:8px 0 0 0; color:white;'>
        Compra: {formatar_data(data_compra_real)} → Venda: {formatar_data(data_venda_real)}
    </p>
</div>
"""
displayHTML(metrics_html)

# COMMAND ----------

# ── Gráfico ───────────────────────────────────────────────────────────────────

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_valores["Date"], y=df_valores["Close"],
    name="Preço", line=dict(color="royalblue")
))
fig.add_trace(go.Scatter(
    x=[dados_compra.iloc[0]["Date"]], y=[close_compra],
    mode="markers", name="Compra",
    marker=dict(color="green", size=14, symbol="triangle-up")
))
fig.add_trace(go.Scatter(
    x=[dados_venda.iloc[0]["Date"]], y=[close_venda],
    mode="markers", name="Venda",
    marker=dict(color="red", size=14, symbol="triangle-down")
))
fig.update_layout(
    title=f"Histórico de Preços — {ticker}",
    xaxis_title="Data", yaxis_title="Preço (R$)",
    hovermode="x unified", template="plotly_dark", height=500
)
displayHTML(fig.to_html())

# COMMAND ----------

# ── Tabela de Valores Recentes ────────────────────────────────────────────────

displayHTML("<h3>Últimos 10 registros</h3>")
display(spark.createDataFrame(df_valores.tail(10)))
