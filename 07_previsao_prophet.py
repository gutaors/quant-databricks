# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 🔮 Previsão com Prophet
# MAGIC Análise preditiva usando Facebook Prophet para prever valores de ações.
# MAGIC
# MAGIC Equivalente a `pages/prophet_arq_local.py` e `pages/prophet.py` do Streamlit.

# COMMAND ----------

# MAGIC %pip install prophet plotly --quiet

# COMMAND ----------

# MAGIC  %restart_python

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

tickers_list = listar_tickers_disponiveis(spark)
dbutils.widgets.dropdown("ticker", tickers_list[0] if tickers_list else "BOVA11.SA", tickers_list, "Selecione a ação")
dbutils.widgets.text("dt_inicial", "2020-01-01", "Data Inicial (AAAA-MM-DD)")
dbutils.widgets.text("dt_final", date.today().strftime("%Y-%m-%d"), "Data Final (AAAA-MM-DD)")
dbutils.widgets.text("meses_previsao", "6", "Meses de Previsão (1-24)")

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
dt_inicial = dbutils.widgets.get("dt_inicial")
dt_final = dbutils.widgets.get("dt_final")
meses = int(dbutils.widgets.get("meses_previsao"))

df = carregar_cotacoes_ticker(spark, ticker)

if df.empty:
    displayHTML(html_alert(f"Nenhum dado para <b>{ticker}</b>.", "error"))
    dbutils.notebook.exit("Sem dados")

# Filtrar por período
df = df.set_index("Date").sort_index()
df = df.loc[(df.index >= pd.to_datetime(dt_inicial)) & (df.index <= pd.to_datetime(dt_final))]
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

if df.empty:
    displayHTML(html_alert("Nenhum dado encontrado no período selecionado.", "warning"))
    dbutils.notebook.exit("Sem dados no período")

displayHTML(f"<h2>Análise Preditiva — {ticker}</h2>")
print(f"Período: {df.index.min().date()} a {df.index.max().date()} | {len(df)} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Dados Históricos

# COMMAND ----------

display(spark.createDataFrame(df.reset_index()))

# COMMAND ----------

# Gráfico de variação no período
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line_color="royalblue"))
fig_hist.update_layout(title=f"Variação no Período — {ticker}", template="plotly_dark", height=400)
displayHTML(fig_hist.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Previsão com Prophet

# COMMAND ----------

# Preparar dados para Prophet
df_prophet = df[["Close"]].copy().reset_index()
df_prophet.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Treinar modelo
modelo = Prophet()
modelo.fit(df_prophet)

# Gerar previsões
datas_futuras = modelo.make_future_dataframe(periods=meses * 30)
previsoes = modelo.predict(datas_futuras)

print(f"Modelo treinado. Previsão gerada para {meses} meses ({meses * 30} dias).")

# COMMAND ----------

# Gráfico de previsão
fig_prev = plot_plotly(modelo, previsoes, xlabel="Período", ylabel="Valor")
fig_prev.update_layout(template="plotly_dark", height=500)
displayHTML(fig_prev.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Tabela de Previsões

# COMMAND ----------

previsoes_display = previsoes[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
previsoes_display["ds"] = previsoes_display["ds"].dt.date
previsoes_display = previsoes_display.sort_values(by="ds", ascending=False)
previsoes_display.columns = ["Data", "Valor_Previsto", "Limite_Inferior", "Limite_Superior"]

display(spark.createDataFrame(previsoes_display.head(60)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Resumo e Recomendação

# COMMAND ----------

ultima_previsao = previsoes.iloc[-1]
valor_previsto = float(ultima_previsao["yhat"])
valor_inferior = float(ultima_previsao["yhat_lower"])
valor_superior = float(ultima_previsao["yhat_upper"])
ultimo_valor_real = float(df["Close"].iloc[-1])

metrics_html = f"""
<div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
    {html_metric("Último Valor Real", formatar_moeda(ultimo_valor_real))}
    {html_metric("Valor Previsto", formatar_moeda(valor_previsto))}
    {html_metric("Intervalo Inferior", formatar_moeda(valor_inferior))}
    {html_metric("Intervalo Superior", formatar_moeda(valor_superior))}
</div>
"""

if valor_previsto > ultimo_valor_real:
    rec_html = html_alert("<b>Recomendação: Tendência Altista</b> — Recomenda-se manter ou aumentar posições de compra.", "success")
else:
    rec_html = html_alert("<b>Recomendação: Tendência Baixista</b> — Recomenda-se cautela ou redução de posições.", "error")

displayHTML(metrics_html + rec_html)

# COMMAND ----------


