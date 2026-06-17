# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Subiu/Dez — Valorização e Desvalorização de 10%
# MAGIC Analisa quais ativos subiram ou desceram 10% ou mais a partir de uma data de referência.
# MAGIC
# MAGIC Equivalente a `pages/subiudez.py` do Streamlit.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
from datetime import datetime

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.text("data_referencia", "2025-01-02", "📅 Data de Referência (AAAA-MM-DD)")

# COMMAND ----------

# ── Configuração ──────────────────────────────────────────────────────────────

LIMIAR = 0.10  # 10%

data_referencia = pd.to_datetime(dbutils.widgets.get("data_referencia"))
tickers = listar_tickers_disponiveis(spark)

if not tickers:
    displayHTML(html_alert("Nenhum ticker encontrado. Importe cotações primeiro.", "warning"))
    dbutils.notebook.exit("Sem tickers")

displayHTML(f"""
<h2>📊 Subiu? — Valorização e Desvalorização de 10%</h2>
<p>Data de referência: <b>{formatar_data(data_referencia)}</b> | Tickers: <b>{len(tickers)}</b></p>
""")

# COMMAND ----------

# ── Análise ───────────────────────────────────────────────────────────────────

def encontrar_cruzamento(df, preco_ref, data_ref, direcao):
    """Busca primeira data após data_ref onde preço cruzou limiar de ±10%."""
    df_futuro = df[df["Date"] > data_ref].copy()
    if df_futuro.empty:
        return None, None

    if direcao == "alta":
        alvo = preco_ref * (1 + LIMIAR)
        mask = df_futuro["Close"] >= alvo
    else:
        alvo = preco_ref * (1 - LIMIAR)
        mask = df_futuro["Close"] <= alvo

    hits = df_futuro[mask]
    if hits.empty:
        return None, None

    primeira = hits.iloc[0]
    return primeira["Date"], float(primeira["Close"])

# COMMAND ----------

resultados_alta = []
resultados_baixa = []

for i, ticker in enumerate(tickers):
    if (i + 1) % 10 == 0:
        print(f"Analisando {i + 1}/{len(tickers)}...")

    df = carregar_cotacoes_ticker(spark, ticker)
    if df.empty:
        continue

    # Encontrar preço na data de referência (ou próxima)
    df_ref = df[df["Date"] >= data_referencia]
    if df_ref.empty:
        continue

    linha_ref = df_ref.iloc[0]
    preco_ref = float(linha_ref["Close"])
    data_efetiva = linha_ref["Date"]
    preco_atual = float(df.iloc[-1]["Close"])

    # Alta ≥ +10%
    data_alta, preco_alta = encontrar_cruzamento(df, preco_ref, data_efetiva, "alta")
    if data_alta is not None:
        variacao = ((preco_alta - preco_ref) / preco_ref) * 100
        dias = (data_alta - data_efetiva).days
        resultados_alta.append({
            "Ticker": ticker,
            "Data_Ref": formatar_data(data_efetiva),
            "Preco_Ref": round(preco_ref, 2),
            "Data_Valorizacao": formatar_data(data_alta),
            "Preco_Valorizacao": round(preco_alta, 2),
            "Variacao_Pct": round(variacao, 2),
            "Dias": dias,
            "Val_Atual": round(preco_atual, 2),
        })

    # Baixa ≥ -10%
    data_baixa, preco_baixa = encontrar_cruzamento(df, preco_ref, data_efetiva, "baixa")
    if data_baixa is not None:
        variacao = ((preco_baixa - preco_ref) / preco_ref) * 100
        dias = (data_baixa - data_efetiva).days
        resultados_baixa.append({
            "Ticker": ticker,
            "Data_Ref": formatar_data(data_efetiva),
            "Preco_Ref": round(preco_ref, 2),
            "Data_Desvalorizacao": formatar_data(data_baixa),
            "Preco_Desvalorizacao": round(preco_baixa, 2),
            "Variacao_Pct": round(variacao, 2),
            "Dias": dias,
            "Val_Atual": round(preco_atual, 2),
        })

print(f"Análise completa: {len(tickers)} tickers processados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🟢 Valorização ≥ +10%

# COMMAND ----------

if resultados_alta:
    df_alta = pd.DataFrame(resultados_alta).sort_values("Dias")
    displayHTML(f"<h3>🟢 Valorização ≥ +10%</h3><p>{len(df_alta)} de {len(tickers)} tickers atingiram +10%</p>")
    display(spark.createDataFrame(df_alta))
else:
    displayHTML(html_alert("Nenhum ticker atingiu valorização de +10% após a data selecionada.", "info"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔴 Desvalorização ≥ -10%

# COMMAND ----------

if resultados_baixa:
    df_baixa = pd.DataFrame(resultados_baixa).sort_values("Dias")
    displayHTML(f"<h3>🔴 Desvalorização ≥ -10%</h3><p>{len(df_baixa)} de {len(tickers)} tickers atingiram -10%</p>")
    display(spark.createDataFrame(df_baixa))
else:
    displayHTML(html_alert("Nenhum ticker atingiu desvalorização de -10% após a data selecionada.", "info"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Resumo

# COMMAND ----------

displayHTML(f"""
<div style='background:#1e1e1e; padding:16px; border-radius:8px; margin:16px 0;'>
    <h3 style='color:white;'>Resumo da análise a partir de {formatar_data(data_referencia)}</h3>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        {html_metric("Tickers Analisados", str(len(tickers)))}
        {html_metric("Valorizaram ≥ +10%", str(len(resultados_alta)), color="#0f9d58")}
        {html_metric("Desvalorizaram ≥ -10%", str(len(resultados_baixa)), color="#dc3545")}
    </div>
</div>
""")
