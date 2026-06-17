# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🏦 Recomendações de Bancos e Agências
# MAGIC Consulta o histórico de recomendações de analistas (Compra/Venda/Manter) via Yahoo Finance.
# MAGIC
# MAGIC Equivalente a `pages/recomendacoes_bancos.py` do Streamlit.

# COMMAND ----------

# MAGIC %pip install yfinance --quiet

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

tickers_list = listar_tickers_disponiveis(spark)
if tickers_list:
    dbutils.widgets.dropdown("ticker", tickers_list[0], tickers_list, "Escolha o Ativo")
else:
    dbutils.widgets.text("ticker", "ITUB4.SA", "Digite o Ticker (ex: ITUB4.SA)")

# COMMAND ----------

# ── Buscar Recomendações ──────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")

# Garantir formato correto do ticker
ticker_busca = ticker
if not ticker_busca.endswith(".SA") and not ticker_busca.endswith(".sa") and len(ticker_busca) >= 4:
    pass  # Mantém como está para suportar tickers internacionais

displayHTML(f"<h2>🏦 Recomendações — {ticker_busca}</h2>")

try:
    ativo = yf.Ticker(ticker_busca)
    df_recomendacoes = ativo.get_recommendations()

    if df_recomendacoes is not None and not df_recomendacoes.empty:
        displayHTML(html_alert(f"Foram encontradas <b>{len(df_recomendacoes)}</b> avaliações para {ticker_busca}.", "success"))

        # Exibir tabela
        display(spark.createDataFrame(df_recomendacoes.reset_index()))

        # Gráfico de barras das recomendações
        colunas_voto = [c for c in df_recomendacoes.columns if c.lower() in ['strongbuy', 'buy', 'hold', 'sell', 'strongsell']]

        if colunas_voto:
            displayHTML("<h3>📊 Visão Geral das Recomendações</h3>")
            
            fig = go.Figure()
            cores = {
                'strongbuy': '#0f9d58', 'buy': '#4caf50',
                'hold': '#ff9800', 'sell': '#f44336', 'strongsell': '#b71c1c'
            }
            
            for col in colunas_voto:
                fig.add_trace(go.Bar(
                    name=col.capitalize(),
                    x=df_recomendacoes.index.astype(str),
                    y=df_recomendacoes[col],
                    marker_color=cores.get(col.lower(), '#666')
                ))
            
            fig.update_layout(
                barmode='group',
                title=f"Recomendações por Período — {ticker_busca}",
                template="plotly_dark",
                height=400,
                xaxis_title="Período",
                yaxis_title="Número de Recomendações"
            )
            displayHTML(fig.to_html())
    else:
        displayHTML(html_alert(f"O Yahoo Finance não retornou nenhuma recomendação para <b>{ticker_busca}</b>.", "warning"))

except Exception as e:
    displayHTML(html_alert(f"Erro ao buscar recomendações: {str(e)}", "error"))
