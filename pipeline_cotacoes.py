# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # ⚡ Pipeline de Cotações Intraday
# MAGIC Coleta automatizada de cotações a cada **30 minutos** durante o pregão B3 (10h–17h).
# MAGIC
# MAGIC Descobre os tickers existentes na tabela `cotacoes_historicas`,
# MAGIC baixa o preço atual de cada um via Yahoo Finance e faz **append**
# MAGIC na tabela `cotacoes_intraday` com a coluna `Hora_Cotacao`.
# MAGIC
# MAGIC **Frequência:** a cada 30 min | **Horário:** Seg–Sex 10h–17h BRT
# MAGIC
# MAGIC **Modos de uso:**
# MAGIC - **Manual**: Execute o notebook pelo Databricks quando quiser uma coleta
# MAGIC - **Agendado**: Configure cron no WSL para executar a cada 30 min via API REST

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
from zoneinfo import ZoneInfo

# COMMAND ----------

# ── Configurações do Pipeline ─────────────────────────────────────────────────

FUSO_SP = ZoneInfo("America/Sao_Paulo")
HORA_ABERTURA = 10   # Pregão B3 abre às 10h
HORA_FECHAMENTO = 17 # Pregão B3 fecha às 17h
DIAS_UTEIS = range(0, 5)  # 0=segunda, 4=sexta

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Verificação de Horário de Pregão

# COMMAND ----------

agora = datetime.now(FUSO_SP)
dia_semana = agora.weekday()  # 0=seg, 6=dom
hora_atual = agora.hour

fora_do_pregao = (dia_semana not in DIAS_UTEIS) or (hora_atual < HORA_ABERTURA) or (hora_atual >= HORA_FECHAMENTO)

if fora_do_pregao:
    msg = (
        f"⏸️ Fora do horário de pregão. "
        f"Agora: {agora.strftime('%A %H:%M')} BRT. "
        f"Pregão: Seg–Sex, {HORA_ABERTURA}h–{HORA_FECHAMENTO}h."
    )
    displayHTML(html_alert(msg, "warning"))
    print(msg)
    dbutils.notebook.exit("Fora do pregão — coleta não realizada")

displayHTML(html_alert(
    f"✅ Dentro do pregão — {agora.strftime('%d/%m/%Y %H:%M:%S')} BRT",
    "success"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Descobrir Tickers Existentes

# COMMAND ----------

tickers = listar_tickers_disponiveis(spark)

if not tickers:
    displayHTML(html_alert(
        "⚠️ Nenhum ticker encontrado na tabela de cotações históricas. "
        "Importe o primeiro ticker usando o notebook <b>03_importa_cotacoes</b>.",
        "warning"
    ))
    dbutils.notebook.exit("Nenhum ticker para coletar")

displayHTML(f"<h2>⚡ Pipeline Intraday — {len(tickers)} tickers</h2>")
displayHTML(f"<p>Tickers encontrados: <b>{', '.join(tickers)}</b></p>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Coletar Cotações Atuais

# COMMAND ----------

def coletar_preco_atual(ticker):
    """
    Baixa o preço mais recente de um ticker via yfinance.
    Retorna um dict com os dados ou None em caso de erro.
    """
    try:
        # Baixa dados do dia atual com intervalo de 1 minuto
        df = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            multi_level_index=False
        )
        if df.empty:
            # Fallback: tenta período de 5 dias (pode ser fim de semana)
            df = yf.download(
                ticker,
                period="5d",
                interval="1m",
                progress=False,
                multi_level_index=False
            )
        if df.empty:
            print(f"  ⚠️ {ticker}: Sem dados disponíveis")
            return None

        # Pega a última linha (preço mais recente)
        ultimo = df.iloc[-1]
        data_pregao = df.index[-1]

        return {
            "Date": data_pregao.date() if hasattr(data_pregao, 'date') else pd.to_datetime(data_pregao).date(),
            "Open": float(ultimo.get("Open", 0)),
            "High": float(ultimo.get("High", 0)),
            "Low": float(ultimo.get("Low", 0)),
            "Close": float(ultimo.get("Close", 0)),
            "Volume": int(ultimo.get("Volume", 0)),
            "Ticker": ticker,
        }

    except Exception as e:
        print(f"  ❌ {ticker}: Erro — {e}")
        return None

# COMMAND ----------

# Momento da coleta (reutiliza FUSO_SP definido na seção de configurações)
hora_coleta = datetime.now(FUSO_SP)

print(f"🕐 Início da coleta: {hora_coleta.strftime('%d/%m/%Y %H:%M:%S')} (BRT)")
print(f"   Coletando {len(tickers)} tickers...\n")

resultados = []
erros = []

for ticker in tickers:
    dados = coletar_preco_atual(ticker)
    if dados:
        dados["Hora_Cotacao"] = hora_coleta.strftime("%Y-%m-%d %H:%M:%S")
        resultados.append(dados)
        print(f"  ✅ {ticker}: R$ {dados['Close']:.2f}")
    else:
        erros.append(ticker)

print(f"\n📊 Coletados: {len(resultados)} | Erros: {len(erros)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Salvar na Tabela Delta (Append)

# COMMAND ----------

if resultados:
    # Criar DataFrame com os resultados
    df_intraday = pd.DataFrame(resultados)

    # Converter tipos
    df_intraday["Date"] = pd.to_datetime(df_intraday["Date"])
    df_intraday["Hora_Cotacao"] = pd.to_datetime(df_intraday["Hora_Cotacao"])
    df_intraday["Open"] = df_intraday["Open"].astype(float)
    df_intraday["High"] = df_intraday["High"].astype(float)
    df_intraday["Low"] = df_intraday["Low"].astype(float)
    df_intraday["Close"] = df_intraday["Close"].astype(float)
    df_intraday["Volume"] = df_intraday["Volume"].astype(int)

    # Converter para Spark DataFrame e salvar como append
    sdf_intraday = spark.createDataFrame(df_intraday)
    sdf_intraday.write.mode("append").saveAsTable(TABELA_COTACOES_INTRADAY)

    displayHTML(html_alert(
        f"✅ <b>{len(resultados)}</b> cotações salvas na tabela "
        f"<code>{TABELA_COTACOES_INTRADAY}</code> "
        f"às <b>{hora_coleta.strftime('%H:%M:%S')}</b> (BRT).",
        "success"
    ))
else:
    displayHTML(html_alert(
        "❌ Nenhuma cotação foi coletada. Verifique os logs acima.",
        "error"
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resumo da Coleta

# COMMAND ----------

if resultados:
    # Exibir resumo em cards
    cards_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>"
    for r in resultados:
        cards_html += html_metric(
            r["Ticker"].replace(".SA", ""),
            formatar_moeda(r["Close"])
        )
    cards_html += "</div>"
    displayHTML(cards_html)

    # Exibir tabela completa
    display(spark.table(TABELA_COTACOES_INTRADAY)
            .filter(F.col("Hora_Cotacao") == hora_coleta.strftime("%Y-%m-%d %H:%M:%S"))
            .orderBy("Ticker"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Histórico de Coletas do Dia

# COMMAND ----------

try:
    hoje = datetime.now(FUSO_SP).strftime("%Y-%m-%d")
    df_hoje = (spark.table(TABELA_COTACOES_INTRADAY)
               .filter(F.col("Date") == hoje)
               .groupBy("Hora_Cotacao")
               .agg(
                   F.count("*").alias("Tickers_Coletados"),
                   F.round(F.avg("Close"), 2).alias("Media_Close")
               )
               .orderBy("Hora_Cotacao"))

    if df_hoje.count() > 0:
        displayHTML("<h3>📅 Coletas realizadas hoje</h3>")
        display(df_hoje)
    else:
        displayHTML(html_alert("Esta é a primeira coleta do dia.", "info"))
except Exception as e:
    displayHTML(html_alert(f"Tabela intraday ainda em construção: {e}", "info"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Estatísticas da Tabela Intraday

# COMMAND ----------

try:
    stats = (spark.table(TABELA_COTACOES_INTRADAY)
             .agg(
                 F.count("*").alias("Total_Registros"),
                 F.countDistinct("Ticker").alias("Total_Tickers"),
                 F.countDistinct("Hora_Cotacao").alias("Total_Coletas"),
                 F.min("Date").alias("Primeira_Data"),
                 F.max("Date").alias("Ultima_Data")
             )
             .collect()[0])

    displayHTML(f"""
    <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
        {html_metric("Total Registros", f"{stats['Total_Registros']:,}")}
        {html_metric("Tickers Monitorados", stats['Total_Tickers'])}
        {html_metric("Coletas Realizadas", stats['Total_Coletas'])}
        {html_metric("Primeira Data", str(stats['Primeira_Data']))}
        {html_metric("Última Data", str(stats['Ultima_Data']))}
    </div>
    """)
except Exception:
    pass

# COMMAND ----------

print(f"\n✅ Pipeline concluído às {datetime.now(FUSO_SP).strftime('%H:%M:%S')} (BRT)")

# COMMAND ----------


