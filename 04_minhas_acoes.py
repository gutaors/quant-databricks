# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 👛 Minhas Ações — Análise do Portfólio
# MAGIC Análises detalhadas das ações do seu portfólio: valores máximos, mínimos, médias móveis.
# MAGIC
# MAGIC Equivalente a `pages/minhas_acoes.py` do Streamlit.

# COMMAND ----------

# MAGIC %pip install yfinance --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

df_tickers = carregar_tickers(spark)

if df_tickers.empty or "ticker" not in df_tickers.columns:
    displayHTML(html_alert("Tabela de tickers do portfólio não encontrada ou vazia.", "error"))
    dbutils.notebook.exit("Sem tickers")

tickers = df_tickers["ticker"].tolist()
displayHTML(f"<h2>Análise de {len(tickers)} ações do portfólio</h2>")
display(spark.createDataFrame(df_tickers))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar Cotações dos Tickers

# COMMAND ----------

# Carregar cotações de todos os tickers do portfólio
frames = []
for ticker in tickers:
    pdf = carregar_cotacoes_ticker(spark, ticker)
    if not pdf.empty:
        if "Ticker" not in pdf.columns:
            pdf["Ticker"] = ticker
        frames.append(pdf)
    else:
        print(f"⚠️ Sem dados para {ticker}")

if not frames:
    displayHTML(html_alert("Nenhum dado de cotação encontrado para os tickers do portfólio. Importe-os primeiro.", "error"))
    dbutils.notebook.exit("Sem cotações")

df_cotacoes = pd.concat(frames, ignore_index=True)
df_cotacoes["Date"] = pd.to_datetime(df_cotacoes["Date"])
df_cotacoes["Close"] = pd.to_numeric(df_cotacoes["Close"], errors="coerce")

print(f"Total de registros carregados: {len(df_cotacoes)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Valores Máximos de Close por Ticker

# COMMAND ----------

# Calcular média móvel de 15 períodos por ticker
df_cotacoes["Media_Movel_15"] = df_cotacoes.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=15).mean()
)

# Encontrar valor máximo com data
def encontrar_valor_maximo_com_data(df):
    return df.groupby("Ticker").apply(lambda x: x.loc[x["Close"].idxmax()]).reset_index(drop=True)

valores_maximos = encontrar_valor_maximo_com_data(df_cotacoes)
displayHTML("<h3>🟢 Valores Máximos de Close por Ticker</h3>")
display(spark.createDataFrame(valores_maximos[["Ticker", "Close", "Date"]].rename(columns={"Close": "Close_Maximo", "Date": "Data_Maximo"})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📉 Valores Mínimos de Close por Ticker

# COMMAND ----------

def encontrar_valor_minimo_com_data(df):
    return df.groupby("Ticker").apply(lambda x: x.loc[x["Close"].idxmin()]).reset_index(drop=True)

valores_minimos = encontrar_valor_minimo_com_data(df_cotacoes)
displayHTML("<h3>🔴 Valores Mínimos de Close por Ticker</h3>")
display(spark.createDataFrame(valores_minimos[["Ticker", "Close", "Date"]].rename(columns={"Close": "Close_Minimo", "Date": "Data_Minimo"})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Média Móvel de 20 dias — Máximos e Mínimos

# COMMAND ----------

# Calcular MM de 20 dias
df_cotacoes["MM20"] = df_cotacoes.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=20).mean()
)

# MM20 máximo por ticker
def encontrar_MM20_max(df):
    return df.dropna(subset=["MM20"]).groupby("Ticker").apply(lambda x: x.loc[x["MM20"].idxmax()]).reset_index(drop=True)

def encontrar_MM20_min(df):
    return df.dropna(subset=["MM20"]).groupby("Ticker").apply(lambda x: x.loc[x["MM20"].idxmin()]).reset_index(drop=True)

mm20_max = encontrar_MM20_max(df_cotacoes)
mm20_min = encontrar_MM20_min(df_cotacoes)

displayHTML("<h3>📈 Valores Máximos da MM20 por Ticker</h3>")
display(spark.createDataFrame(mm20_max[["Ticker", "MM20", "Date"]].rename(columns={"MM20": "MM20_Max", "Date": "Data_MM20_Max"})))

displayHTML("<h3>📉 Valores Mínimos da MM20 por Ticker</h3>")
display(spark.createDataFrame(mm20_min[["Ticker", "MM20", "Date"]].rename(columns={"MM20": "MM20_Min", "Date": "Data_MM20_Min"})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔗 Junção — Análise Consolidada

# COMMAND ----------

# Juntar tickers com valores máximos
df_analises = df_tickers.merge(
    valores_maximos[["Ticker", "Close", "Date"]].rename(columns={"Close": "Close_Max", "Date": "Data_Max", "Ticker": "ticker"}),
    on="ticker", how="left"
)

df_analises = df_analises.merge(
    valores_minimos[["Ticker", "Close", "Date"]].rename(columns={"Close": "Close_Min", "Date": "Data_Min", "Ticker": "ticker"}),
    on="ticker", how="left"
)

displayHTML("<h3>📋 Análise Consolidada do Portfólio</h3>")
display(spark.createDataFrame(df_analises))

# COMMAND ----------


