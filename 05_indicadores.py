# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 📉 Indicadores Técnicos — MM200, RSI, MACD
# MAGIC Análise técnica completa: Média Móvel 200, RSI (14 dias), MACD com alertas e recomendações.
# MAGIC
# MAGIC Equivalente a `pages/graficos_arq_local.py` e seção "Indicadores" de `consolidado_local.py`.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
from datetime import date

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown("ticker", "", listar_tickers_disponiveis(spark), "Escolha o Ativo")
dbutils.widgets.text("data_corte", date.today().strftime("%Y-%m-%d"), "Data de Corte (AAAA-MM-DD)")

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
cut_date = pd.to_datetime(dbutils.widgets.get("data_corte"))

df = carregar_cotacoes_ticker(spark, ticker)

if df.empty:
    displayHTML(html_alert(f"Nenhum dado encontrado para <b>{ticker}</b>.", "error"))
    dbutils.notebook.exit("Sem dados")

# Preparar DataFrame
df = df.set_index("Date").sort_index()
df_cut = df[df.index <= cut_date].copy()
df_cut["Close"] = pd.to_numeric(df_cut["Close"], errors="coerce")
df_cut = df_cut.dropna(subset=["Close"])

displayHTML(f"""
<h2>Indicadores Técnicos — {ticker}</h2>
<p>Dados filtrados até {formatar_data(cut_date)} | {len(df_cut)} registros</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cálculo dos Indicadores

# COMMAND ----------

if df_cut.empty or "Close" not in df_cut.columns:
    displayHTML(html_alert("Coluna 'Close' não encontrada ou sem dados.", "error"))
    dbutils.notebook.exit("Sem Close")

# MM200
df_cut["MM200"] = df_cut["Close"].rolling(window=200).mean()

# RSI (14 dias)
delta = df_cut["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df_cut["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df_cut["Close"].ewm(span=12, adjust=False).mean()
ema26 = df_cut["Close"].ewm(span=26, adjust=False).mean()
df_cut["MACD"] = ema12 - ema26
df_cut["Signal"] = df_cut["MACD"].ewm(span=9, adjust=False).mean()

print("Indicadores calculados: MM200, RSI, MACD")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Gráficos de Indicadores

# COMMAND ----------

import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=("Preço + MM200", "RSI (14)", "MACD"),
    vertical_spacing=0.08, row_heights=[0.5, 0.25, 0.25]
)

# Preço + MM200
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Close"], name="Close", line_color="royalblue"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM200"], name="MM200", line_color="orange"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["RSI"], name="RSI", line_color="purple"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MACD"], name="MACD", line_color="cyan"), row=3, col=1)
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Signal"], name="Signal", line_color="magenta"), row=3, col=1)

fig.update_layout(title=f"Indicadores Técnicos — {ticker}", template="plotly_dark", height=900)
displayHTML(fig.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚨 Alertas e Recomendações

# COMMAND ----------

tol = 0.01
alertas_html = ""

if len(df_cut) >= 2:
    current = df_cut.iloc[-1]
    previous = df_cut.iloc[-2]

    tocou_de_cima = previous["Close"] > previous["MM200"] and abs(current["Close"] - current["MM200"]) / current["MM200"] < tol
    tocou_de_baixo = previous["Close"] < previous["MM200"] and abs(current["Close"] - current["MM200"]) / current["MM200"] < tol

    # Alerta 1
    if tocou_de_cima:
        alertas_html += html_alert("Alerta 1: Preço do ativo <b>caiu para a MM200</b>. Fique atento para oportunidades de compra.", "info")
        if current["RSI"] > 50 and current["MACD"] > current["Signal"]:
            alertas_html += html_alert("Alerta 2: <b>Suporte forte</b> — Em tendência de alta, o toque na MM200 pode indicar bom ponto de compra.", "success")

    # Alerta 3
    if previous["Close"] > previous["MM200"] and current["Close"] < current["MM200"]:
        alertas_html += html_alert("Alerta 3: <b>Rompimento baixista</b> — O preço caiu abaixo da MM200. Considere reduzir posições.", "error")

    # Alerta 4
    if tocou_de_baixo:
        alertas_html += html_alert("Alerta 4: Preço do ativo <b>subiu para a MM200</b>. Possível resistência.", "warning")
        if current["RSI"] < 50 and current["MACD"] < current["Signal"]:
            alertas_html += html_alert("Alerta 5: <b>Resistência forte</b> — Evite compras, considere proteção.", "error")

    # Alerta 6
    if previous["Close"] < previous["MM200"] and current["Close"] > current["MM200"]:
        alertas_html += html_alert("Alerta 6: <b>Rompimento altista</b> — O preço superou a MM200. Oportunidade de compra.", "success")

    # Resumo dos indicadores
    alertas_html += f"""
    <div style='background:#1e1e1e; padding:16px; border-radius:8px; margin:16px 0;'>
        <h3 style='color:white;'>📋 Resumo dos Indicadores</h3>
        <div style='display:flex; flex-wrap:wrap; gap:8px;'>
            {html_metric("Preço Atual", f"{current['Close']:.2f}")}
            {html_metric("MM200", f"{current['MM200']:.2f}")}
            {html_metric("RSI", f"{current['RSI']:.2f}")}
            {html_metric("MACD", f"{current['MACD']:.2f}")}
            {html_metric("MACD Signal", f"{current['Signal']:.2f}")}
        </div>
    </div>
    """

    # Recomendação geral
    if current["RSI"] > 50 and current["MACD"] > current["Signal"] and current["Close"] > current["MM200"]:
        alertas_html += html_alert("<b>Tendência Altista:</b> Recomenda-se manter ou aumentar posições de compra.", "success")
    elif current["RSI"] < 50 and current["MACD"] < current["Signal"] and current["Close"] < current["MM200"]:
        alertas_html += html_alert("<b>Tendência Baixista:</b> Recomenda-se cautela ou redução de posições.", "error")
    else:
        alertas_html += html_alert("<b>Sinal Neutro/Misto:</b> Aguarde confirmações adicionais.", "warning")

displayHTML(alertas_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Volume e Dividendos

# COMMAND ----------

charts_html = ""

if "Volume" in df_cut.columns:
    df_cut["Volume"] = pd.to_numeric(df_cut["Volume"], errors="coerce")
    vol = df_cut["Volume"].dropna()
    if not vol.empty:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=vol.index, y=vol.values, name="Volume", marker_color="rgba(100,149,237,0.6)"))
        fig_vol.update_layout(title="Volume de Negociação", template="plotly_dark", height=300)
        displayHTML(fig_vol.to_html())

if "Dividends" in df_cut.columns:
    df_cut["Dividends"] = pd.to_numeric(df_cut["Dividends"], errors="coerce")
    div = df_cut["Dividends"].dropna()
    div_nonzero = div[div > 0]
    if not div_nonzero.empty:
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(x=div_nonzero.index, y=div_nonzero.values, name="Dividendos", marker_color="rgba(46,125,50,0.7)"))
        fig_div.update_layout(title="Dividendos", template="plotly_dark", height=300)
        displayHTML(fig_div.to_html())
    else:
        print("Sem dados de dividendos não-zero.")
else:
    print("Coluna 'Dividends' não disponível.")
