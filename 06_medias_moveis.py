# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Médias Móveis — Cruzamentos MM50 x MM200
# MAGIC Detecção de cruzamentos entre médias móveis de 50 e 200 períodos com sinais de compra/venda,
# MAGIC backtesting e análise de concretização.
# MAGIC
# MAGIC Equivalente a `pages/mediasmoveis.py` do Streamlit.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

tickers_list = listar_tickers_disponiveis(spark)
dbutils.widgets.dropdown("ticker", tickers_list[0] if tickers_list else "", tickers_list, "Escolha o Ativo")
dbutils.widgets.text("data_corte", date.today().strftime("%Y-%m-%d"), "Data de Corte (AAAA-MM-DD)")

displayHTML("<b>Escolha o Ativo:</b>")
displayHTML(dbutils.widgets.get("ticker"))
displayHTML("<b>Data de Corte:</b>")
displayHTML(dbutils.widgets.get("data_corte"))

# COMMAND ----------

# ── Carregar e Preparar Dados ─────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
cut_date = pd.to_datetime(dbutils.widgets.get("data_corte"))

df = carregar_cotacoes_ticker(spark, ticker)

if df.empty:
    displayHTML(html_alert(f"Nenhum dado para <b>{ticker}</b>.", "error"))
    dbutils.notebook.exit("Sem dados")

df = df.set_index("Date").sort_index()
df_cut = df[df.index <= cut_date].copy()
df_cut["Close"] = pd.to_numeric(df_cut["Close"], errors="coerce")
df_cut = df_cut.dropna(subset=["Close"])

if "Volume" in df_cut.columns:
    df_cut["Volume"] = pd.to_numeric(df_cut["Volume"], errors="coerce")

displayHTML(f"<h2>Médias Móveis — {ticker}</h2>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dados mais recentes e extremos

# COMMAND ----------

if not df_cut.empty:
    ultimo = df_cut.tail(1)
    ultimo_preco = float(ultimo["Close"].iloc[0])
    max_idx = df_cut["Close"].idxmax()
    min_idx = df_cut["Close"].idxmin()
    max_val = float(df_cut.loc[max_idx, "Close"])
    min_val = float(df_cut.loc[min_idx, "Close"])

    metrics = f"""
    <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
        {html_metric("Último Preço", formatar_moeda(ultimo_preco), f"Data: {formatar_data(ultimo.index[0])}")}
        {html_metric("Valor Máximo", formatar_moeda(max_val), f"Em {formatar_data(max_idx)}", "#0f9d58")}
        {html_metric("Valor Mínimo", formatar_moeda(min_val), f"Em {formatar_data(min_idx)}", "#dc3545")}
    """
    if "Volume" in ultimo.columns and pd.notna(ultimo["Volume"].iloc[0]):
        ultimo_vol = int(ultimo["Volume"].iloc[0])
        metrics += html_metric("Volume", f"{ultimo_vol:,}")
    metrics += "</div>"
    displayHTML(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Gráfico MM50 x MM200

# COMMAND ----------

df_cut["MM50"] = df_cut["Close"].rolling(window=50).mean()
df_cut["MM200"] = df_cut["Close"].rolling(window=200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Close"], name="Close", line_color="royalblue"))
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM50"], name="MM50", line_color="orange"))
fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM200"], name="MM200", line_color="green"))
fig.update_layout(title=f"Fechamento com MM50 e MM200 — {ticker}", template="plotly_dark", height=500)
displayHTML(fig.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔀 Pontos de Cruzamento

# COMMAND ----------

# Detectar cruzamentos
cross = df_cut["MM50"] > df_cut["MM200"]
sinal = cross.ne(cross.shift())
cross_df = df_cut[sinal].copy()
cross_df = cross_df.dropna(subset=["MM50", "MM200"])

if not cross_df.empty:
    cross_df["Recomendacao"] = cross_df.apply(
        lambda x: "COMPRA" if x["MM50"] > x["MM200"] else "VENDA", axis=1
    )

    tabela = pd.DataFrame({
        "No": range(1, len(cross_df) + 1),
        "Data": cross_df.index.strftime("%d/%m/%Y"),
        "Preco": [f"R$ {p:.2f}" for p in cross_df["Close"]],
        "Recomendacao": cross_df["Recomendacao"]
    })

    displayHTML("<h3>Pontos de Cruzamento das Médias Móveis</h3>")
    display(spark.createDataFrame(tabela))
else:
    displayHTML(html_alert("Nenhum cruzamento encontrado no período.", "info"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💰 Backtesting — Resultado com R$ 10.000

# COMMAND ----------

if not cross_df.empty:
    df_bt = cross_df[["Close", "Recomendacao"]].copy()
    df_bt = df_bt.sort_index()

    valor = 10000.0
    quantidade = 0.0
    comprado = False

    for idx, row in df_bt.iterrows():
        preco = float(row["Close"])
        rec = row["Recomendacao"]
        if rec == "COMPRA" and not comprado:
            quantidade = valor / preco
            comprado = True
        elif rec == "VENDA" and comprado:
            valor = quantidade * preco
            quantidade = 0
            comprado = False

    preco_ultimo = float(df_bt["Close"].iloc[-1])
    valor_final = valor if not comprado else quantidade * preco_ultimo
    lucro = valor_final - 10000.0
    cor = "#0f9d58" if lucro >= 0 else "#dc3545"

    displayHTML(f"""
    <div style='background:{cor}; color:white; padding:20px; border-radius:10px; text-align:center; margin:16px 0;'>
        <h2 style='margin:0; color:white;'>Backtesting: R$ 10.000 → {formatar_moeda(valor_final)}</h2>
        <p style='margin:8px 0 0 0; color:white;'>Lucro/Prejuízo: {formatar_moeda(lucro)} | Status: {"Comprado" if comprado else "Vendido"}</p>
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⏱️ Concretização das Recomendações (±10%)

# COMMAND ----------

if not cross_df.empty:
    dados_conc = []
    numero = 1

    for idx, row in cross_df.iterrows():
        preco_ref = float(row["Close"])
        data_ref = idx
        rec = row["Recomendacao"]

        precos_futuros = df_cut.loc[df_cut.index > data_ref, "Close"]

        if rec == "COMPRA":
            alvo = preco_ref * 1.10
            mask = precos_futuros >= alvo
        else:
            alvo = preco_ref * 0.90
            mask = precos_futuros <= alvo

        if mask.any():
            data_conc = precos_futuros[mask].index[0]
            preco_conc = float(precos_futuros[mask].iloc[0])
            dias = (data_conc - data_ref).days

            dados_conc.append({
                "No": numero,
                "Data_Sinal": formatar_data(data_ref),
                "Preco_Sinal": formatar_moeda(preco_ref),
                "Recomendacao": rec,
                "Data_Concretizacao": formatar_data(data_conc),
                "Dias_Passados": dias,
                "Preco_Concretizacao": formatar_moeda(preco_conc)
            })
            numero += 1

    if dados_conc:
        df_conc = pd.DataFrame(dados_conc)
        displayHTML("<h3>Data da Concretização (±10%)</h3>")
        display(spark.createDataFrame(df_conc))

        # Resumo
        df_compras = df_conc[df_conc["Recomendacao"] == "COMPRA"]
        df_vendas = df_conc[df_conc["Recomendacao"] == "VENDA"]

        media_compra = df_compras["Dias_Passados"].mean() if not df_compras.empty else 0
        media_venda = df_vendas["Dias_Passados"].mean() if not df_vendas.empty else 0
        mediana_compra = df_compras["Dias_Passados"].median() if not df_compras.empty else 0
        mediana_venda = df_vendas["Dias_Passados"].median() if not df_vendas.empty else 0

        displayHTML(f"""
        <div style='display:flex; gap:32px; margin:16px 0;'>
            <div>
                <h4 style='color:#0f9d58;'>Dias para Valorização +10%</h4>
                <p style='color:#0f9d58; font-size:20px;'><b>Média: {media_compra:.0f} dias</b></p>
                <p style='color:#0f9d58; font-size:20px;'><b>Mediana: {mediana_compra:.0f} dias</b></p>
            </div>
            <div>
                <h4 style='color:#dc3545;'>Dias para Desvalorização -10%</h4>
                <p style='color:#dc3545; font-size:20px;'><b>Média: {media_venda:.0f} dias</b></p>
                <p style='color:#dc3545; font-size:20px;'><b>Mediana: {mediana_venda:.0f} dias</b></p>
            </div>
        </div>
        """)
    else:
        displayHTML(html_alert("Nenhuma concretização encontrada.", "info"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Pesquisa de Preços por Data

# COMMAND ----------

dbutils.widgets.text("data_pesquisa", date.today().strftime("%Y-%m-%d"), "Data para pesquisar preços (AAAA-MM-DD)")

# COMMAND ----------

data_pesq = pd.to_datetime(dbutils.widgets.get("data_pesquisa"))
df_pesq = df_cut[df_cut.index >= data_pesq].head(21)

if not df_pesq.empty:
    result = pd.DataFrame({
        "Data": df_pesq.index.strftime("%d/%m/%Y"),
        "Preco_Fechamento": [formatar_moeda(p) for p in df_pesq["Close"]]
    })
    displayHTML(f"<h3>Preços — 20 dias a partir de {formatar_data(data_pesq)}</h3>")
    display(spark.createDataFrame(result))
else:
    displayHTML(html_alert("Sem dados para o período.", "warning"))

# COMMAND ----------

displayHTML("<hr/><p>Desenvolvido por Gustavo</p>")
