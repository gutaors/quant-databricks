# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 📥 Importa Cotações
# MAGIC Baixa cotações do Yahoo Finance e salva na tabela Delta `cotacoes_historicas`.
# MAGIC
# MAGIC Equivalente a `pages/importa_cotacoes.py` do Streamlit.

# COMMAND ----------

# MAGIC %pip install yfinance --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.removeAll()
dbutils.widgets.text("ticker_importar", "", "Ticker para importar (ex: PETR4.SA)")
dbutils.widgets.dropdown("acao", "importar", ["importar", "atualizar_todos", "listar"], "Ação")
displayHTML("<div style='margin-top:20px'></div>")

# COMMAND ----------

acao = dbutils.widgets.get("acao")
ticker_input = dbutils.widgets.get("ticker_importar").strip()

# COMMAND ----------

# ── Funções ───────────────────────────────────────────────────────────────────

def formatar_ticker(ticker):
    """Garante que o ticker termina com .SA."""
    ticker = ticker.strip().upper()
    if not ticker.endswith(".SA"):
        ticker += ".SA"
    return ticker

def verificar_ticker_real(ticker):
    """Verifica se o ticker é válido no Yahoo Finance."""
    try:
        df = yf.download(ticker, period="1d", progress=False)
        return not df.empty
    except Exception:
        return False

def baixar_cotacoes_completo(ticker):
    """Baixa histórico completo de cotações desde 2009."""
    data_inicio = "2009-01-01"
    data_fim = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=data_inicio, end=data_fim, progress=False, multi_level_index=False).reset_index()
    except TypeError:
        df = yf.download(ticker, start=data_inicio, end=data_fim, progress=False).reset_index()
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}")
    df["Ticker"] = ticker
    return df

def atualizar_incremental(spark, ticker):
    """Atualiza incrementalmente as cotações de um ticker."""
    # Verificar última data existente
    try:
        sdf = spark.table(TABELA_COTACOES).filter(f"Ticker = '{ticker}'")
        if sdf.count() == 0:
            raise Exception("Sem dados")
        ultima_data = sdf.agg({"Date": "max"}).collect()[0][0]
        proxima_data = (pd.to_datetime(ultima_data) + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        proxima_data = "2009-01-01"

    data_fim = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    if proxima_data > data_fim:
        print(f"  {ticker}: já atualizado")
        return 0
    
    try:
        df_novo = yf.download(ticker, start=proxima_data, end=data_fim, progress=False, multi_level_index=False).reset_index()
    except TypeError:
        df_novo = yf.download(ticker, start=proxima_data, end=data_fim, progress=False).reset_index()
    
    if df_novo.empty:
        print(f"  {ticker}: sem novos dados entre {proxima_data} e {data_fim}")
        return 0
    
    df_novo["Ticker"] = ticker
    sdf_novo = spark.createDataFrame(df_novo)
    sdf_novo.write.mode("append").saveAsTable(TABELA_COTACOES)
    print(f"  {ticker}: +{len(df_novo)} novos registros")
    return len(df_novo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execução

# COMMAND ----------

if acao == "listar":
    # ── Listar tickers existentes ─────────────────────────────────────────────
    displayHTML("<h2>📋 Tickers Disponíveis na Tabela Delta</h2>")
    try:
        df_lista = (spark.table(TABELA_COTACOES)
                    .groupBy("Ticker")
                    .agg(
                        F.count("*").alias("Registros"),
                        F.min("Date").alias("Data_Inicio"),
                        F.max("Date").alias("Data_Fim")
                    )
                    .orderBy("Ticker"))
        display(df_lista)
    except Exception as e:
        displayHTML(html_alert(f"Tabela {TABELA_COTACOES} não encontrada. Importe o primeiro ticker para criá-la.", "warning"))

elif acao == "importar":
    # ── Importar ticker individual ────────────────────────────────────────────
    if not ticker_input:
        displayHTML(html_alert("Por favor, preencha o campo 'Ticker para importar'.", "warning"))
    else:
        ticker = formatar_ticker(ticker_input)
        displayHTML(f"<h2>📥 Importando: {ticker}</h2>")

        # Verificar se já existe
        try:
            existente = spark.table(TABELA_COTACOES).filter(f"Ticker = '{ticker}'").count()
        except Exception:
            existente = 0

        if existente > 0:
            displayHTML(html_alert(f"O ticker <b>{ticker}</b> já possui {existente} registros. Use 'atualizar_todos' para atualizar.", "warning"))
        else:
            if verificar_ticker_real(ticker):
                df = baixar_cotacoes_completo(ticker)
                sdf = spark.createDataFrame(df)
                sdf.write.mode("append").saveAsTable(TABELA_COTACOES)
                displayHTML(html_alert(f"✅ Ticker <b>{ticker}</b> importado com sucesso! {len(df)} registros.", "success"))
            else:
                displayHTML(html_alert(f"O ticker <b>{ticker}</b> não foi encontrado no Yahoo Finance.", "error"))

elif acao == "atualizar_todos":
    # ── Atualizar todos os tickers existentes ─────────────────────────────────
    displayHTML("<h2>🔄 Atualizando todos os tickers</h2>")
    tickers = listar_tickers_disponiveis(spark)
    
    if not tickers:
        displayHTML(html_alert("Nenhum ticker encontrado. Importe o primeiro ticker usando a ação 'importar'.", "warning"))
    else:
        total_novos = 0
        for t in tickers:
            novos = atualizar_incremental(spark, t)
            total_novos += novos
        displayHTML(html_alert(f"✅ Atualização concluída! {total_novos} novos registros adicionados para {len(tickers)} tickers.", "success"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizar Dados Recentes

# COMMAND ----------

try:
    tickers_existentes = listar_tickers_disponiveis(spark)
    if tickers_existentes:
        for t in tickers_existentes[:5]:  # Mostra até 5 tickers
            print(f"\n── {t} ──")
            pdf = carregar_cotacoes_ticker(spark, t)
            if not pdf.empty:
                print(f"  Primeiro: {pdf['Date'].min().date()}  |  Último: {pdf['Date'].max().date()}  |  Registros: {len(pdf)}")
except Exception:
    print("Tabela ainda não criada. Importe o primeiro ticker.")

# COMMAND ----------


