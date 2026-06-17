# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Utilitários Compartilhados
# MAGIC Funções auxiliares usadas em todos os notebooks do projeto Quant.
# MAGIC
# MAGIC **Execute este notebook antes dos demais** ou use `%run ./utils_quant` nos outros notebooks.

# COMMAND ----------

from datetime import date, datetime, timedelta
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd

# COMMAND ----------

# ── Constantes ────────────────────────────────────────────────────────────────

DATA_INICIO = "2015-01-05"
DATA_FIM = date.today().strftime("%Y-%m-%d")

# Tabelas Delta no Databricks
TABELA_COTACOES = "workspace.default.cotacoes_historicas"
TABELA_TICKERS = "workspace.default.tickers"
TABELA_ACOES = "workspace.default.acoes_raw"
TABELA_ORGAOS = "workspace.default.orgaos_unificados"
TABELA_SIORG = "workspace.default.siorg_unidades"
TABELA_NOTICIAS = "workspace.default.noticias"

# COMMAND ----------

# ── Formatação ────────────────────────────────────────────────────────────────

def formatar_moeda(valor):
    """Formata um valor numérico para o formato de moeda brasileira."""
    return f"R$ {valor:.2f}"

def formatar_data(data):
    """Formata uma data para o padrão brasileiro dd/mm/aaaa."""
    if isinstance(data, str):
        data = pd.to_datetime(data)
    return data.strftime("%d/%m/%Y")

def formatar_valor(valor):
    """Formata valor numérico com separador de milhar BR."""
    return "{:,.2f}".format(valor).replace(",", "").replace(".", ",")

# COMMAND ----------

# ── Helpers de Dados ──────────────────────────────────────────────────────────

def listar_tickers_disponiveis(spark):
    """
    Lista todos os tickers distintos disponíveis na tabela de cotações.
    Retorna lista de strings.
    """
    try:
        df = spark.table(TABELA_COTACOES).select("Ticker").distinct().orderBy("Ticker")
        return [row.Ticker for row in df.collect()]
    except Exception as e:
        print(f"Erro ao listar tickers: {e}")
        return []

def carregar_cotacoes_ticker(spark, ticker):
    """
    Carrega as cotações de um ticker específico da tabela Delta como Pandas DataFrame.
    Colunas esperadas: Date, Open, High, Low, Close, Volume, Ticker
    """
    try:
        sdf = (spark.table(TABELA_COTACOES)
               .filter(F.col("Ticker") == ticker)
               .orderBy("Date"))
        pdf = sdf.toPandas()
        if "Date" in pdf.columns:
            pdf["Date"] = pd.to_datetime(pdf["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in pdf.columns:
                pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        return pdf
    except Exception as e:
        print(f"Erro ao carregar cotações de {ticker}: {e}")
        return pd.DataFrame()

def carregar_tickers(spark):
    """
    Carrega a tabela de tickers (portfólio do usuário) como Pandas DataFrame.
    """
    try:
        return spark.table(TABELA_TICKERS).toPandas()
    except Exception as e:
        print(f"Erro ao carregar tickers: {e}")
        return pd.DataFrame()

def carregar_acoes(spark):
    """
    Carrega a tabela de ações (catálogo de nomes/siglas) como Pandas DataFrame.
    """
    try:
        return spark.table(TABELA_ACOES).toPandas()
    except Exception as e:
        print(f"Erro ao carregar ações: {e}")
        return pd.DataFrame()

def salvar_cotacoes_delta(spark, pdf, ticker, modo="append"):
    """
    Salva um Pandas DataFrame de cotações na tabela Delta.
    Adiciona a coluna Ticker se não existir.
    """
    if "Ticker" not in pdf.columns:
        pdf["Ticker"] = ticker
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode(modo).saveAsTable(TABELA_COTACOES)

# COMMAND ----------

# ── HTML Helpers para Databricks ──────────────────────────────────────────────

def html_title(text, level=1):
    """Exibe um título HTML no notebook."""
    displayHTML(f"<h{level}>{text}</h{level}>")

def html_metric(label, value, delta=None, color=None):
    """Exibe uma métrica estilo card."""
    delta_html = ""
    if delta is not None:
        delta_color = color or ("#0f9d58" if float(str(delta).replace("R$","").replace(",",".").replace("%","")) >= 0 else "#dc3545")
        delta_html = f"<p style='color:{delta_color}; margin:0;'>{delta}</p>"
    html = f"""
    <div style='background:#1e1e1e; border-radius:8px; padding:16px; margin:4px; display:inline-block; min-width:180px;'>
        <p style='color:#aaa; margin:0 0 4px 0; font-size:13px;'>{label}</p>
        <p style='color:#fff; margin:0; font-size:24px; font-weight:bold;'>{value}</p>
        {delta_html}
    </div>
    """
    return html

def html_alert(text, tipo="info"):
    """Exibe um alerta colorido. tipo: info, success, warning, error."""
    cores = {
        "info": ("#e3f2fd", "#1565c0"),
        "success": ("#e8f5e9", "#2e7d32"),
        "warning": ("#fff3e0", "#e65100"),
        "error": ("#ffebee", "#c62828"),
    }
    bg, fg = cores.get(tipo, cores["info"])
    return f"<div style='background:{bg}; color:{fg}; padding:12px; border-radius:6px; margin:8px 0;'>{text}</div>"

# COMMAND ----------

print("✅ utils_quant carregado com sucesso!")
