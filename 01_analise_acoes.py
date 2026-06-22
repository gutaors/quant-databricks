# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Análise de Ações — Painel Principal
# MAGIC Simulador de compra/venda, painel resumido de 5 anos e análise do portfólio.
# MAGIC
# MAGIC Equivalente ao `app.py` do Streamlit.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import plotly.graph_objs as go
from datetime import date, datetime, timedelta

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

tickers_list = listar_tickers_disponiveis(spark)
default_ticker = tickers_list[0] if tickers_list else ""
dbutils.widgets.dropdown("ticker", default_ticker, tickers_list, "Escolha a ação")
dbutils.widgets.text("data_compra", date.today().strftime("%Y-%m-%d"), "Data de Compra (AAAA-MM-DD)")
dbutils.widgets.text("data_venda", date.today().strftime("%Y-%m-%d"), "Data de Venda (AAAA-MM-DD)")
dbutils.widgets.text("valor_investido", "1000.00", "Valor Investido (R$)")

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
data_compra_str = dbutils.widgets.get("data_compra")
data_venda_str = dbutils.widgets.get("data_venda")
valor_sim = float(dbutils.widgets.get("valor_investido"))

data_compra_sim = pd.to_datetime(data_compra_str).date()
data_venda_sim = pd.to_datetime(data_venda_str).date()

df_valores = carregar_cotacoes_ticker(spark, ticker)
df_tickers = carregar_tickers(spark)

if df_valores.empty:
    displayHTML(html_alert(f"Nenhum dado encontrado para o ticker <b>{ticker}</b>. Importe cotações primeiro.", "error"))
    dbutils.notebook.exit("Sem dados")

print(f"Dados carregados: {len(df_valores)} registros para {ticker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Simulador de Compra e Venda

# COMMAND ----------

def obter_dados_acao(data_alvo, df_valores):
    """Obtém dados da ação para uma data, retrocedendo até encontrar dados válidos."""
    data_atual = pd.to_datetime(data_alvo)
    for _ in range(30):
        filtro = df_valores["Date"].dt.date == data_atual.date()
        dados = df_valores[filtro]
        if not dados.empty:
            return dados
        data_atual -= timedelta(days=1)
    return pd.DataFrame()

# Simulação de Compra
dados_compra = obter_dados_acao(data_compra_sim, df_valores)

if not dados_compra.empty:
    close_compra = float(dados_compra.iloc[0]["Close"])
    quantidade_comprada = valor_sim / close_compra

    # Simulação de Venda
    dados_venda = obter_dados_acao(data_venda_sim, df_valores)

    if not dados_venda.empty:
        close_venda = float(dados_venda.iloc[0]["Close"])
        valor_vendido = quantidade_comprada * close_venda
        lucro_prejuizo = valor_vendido - valor_sim
        cor = "#0f9d58" if lucro_prejuizo >= 0 else "#dc3545"
        pct = (lucro_prejuizo / valor_sim) * 100

        metrics_html = f"""
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
            {html_metric("Ação", ticker)}
            {html_metric("Preço Compra", formatar_moeda(close_compra))}
            {html_metric("Preço Venda", formatar_moeda(close_venda))}
            {html_metric("Quantidade", f"{quantidade_comprada:.2f}")}
            {html_metric("Valor Investido", formatar_moeda(valor_sim))}
            {html_metric("Valor Final", formatar_moeda(valor_vendido))}
        </div>
        <div style='background:{cor}; color:white; padding:16px; border-radius:8px; text-align:center; margin:16px 0;'>
            <h2 style='margin:0; color:white;'>Resultado: {formatar_moeda(lucro_prejuizo)} ({pct:.2f}%)</h2>
            <p style='margin:4px 0 0 0; color:white;'>Compra: {formatar_data(data_compra_sim)} → Venda: {formatar_data(data_venda_sim)}</p>
        </div>
        """
        displayHTML(metrics_html)
    else:
        displayHTML(html_alert(f"Não há dados para a data de venda: {formatar_data(data_venda_sim)}", "warning"))
else:
    displayHTML(html_alert(f"Não há dados para a data de compra: {formatar_data(data_compra_sim)}", "warning"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Gráfico de Preços com Médias Móveis

# COMMAND ----------

# Calcular médias móveis
df_valores["MA15"] = df_valores["Close"].rolling(window=15).mean()
df_valores["MA50"] = df_valores["Close"].rolling(window=50).mean()
df_valores["Cruzamento"] = df_valores["MA15"] - df_valores["MA50"]

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df_valores["Date"], y=df_valores["Close"],
                             name="Preço de Fechamento", line_color="royalblue"))
fig_ma.add_trace(go.Scatter(x=df_valores["Date"], y=df_valores["MA15"],
                             name="Média Móvel 15", line_color="orange"))
fig_ma.add_trace(go.Scatter(x=df_valores["Date"], y=df_valores["MA50"],
                             name="Média Móvel 50", line_color="green"))

# Adicionar setas de cruzamento
for i in range(1, len(df_valores)):
    if (df_valores.iloc[i]["Cruzamento"] * df_valores.iloc[i - 1]["Cruzamento"]) < 0:
        arrow_color = "green" if df_valores.iloc[i]["Cruzamento"] > 0 else "red"
        fig_ma.add_annotation(
            x=df_valores.iloc[i]["Date"], y=df_valores.iloc[i]["Close"],
            showarrow=True, arrowhead=1, arrowsize=2, arrowwidth=2, arrowcolor=arrow_color
        )

fig_ma.update_layout(
    title=f"Preços e Médias Móveis — {ticker}",
    template="plotly_dark", height=500,
    xaxis_title="Data", yaxis_title="Preço (R$)"
)
displayHTML(fig_ma.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Painel Resumido — Últimos 5 Anos

# COMMAND ----------

data_atual = pd.to_datetime(date.today())
data_cinco_anos = data_atual - timedelta(days=5 * 365)

df_periodo = df_valores[(df_valores["Date"] >= data_cinco_anos) & (df_valores["Date"] <= data_atual)]

if not df_periodo.empty:
    valor_maximo = float(df_periodo["Close"].max())
    valor_minimo = float(df_periodo["Close"].min())
    valor_medio = float(df_periodo["Close"].mean())
    variancia = float(df_periodo["Close"].var())
    valor_inicio = float(df_periodo["Close"].iloc[0])
    valor_final = float(df_periodo["Close"].iloc[-1])
    dt_inicio = df_periodo["Date"].iloc[0]
    dt_final = df_periodo["Date"].iloc[-1]

    # Encontrar valor mais próximo do atual
    diffs = {
        "Valor Mínimo": abs(valor_final - valor_minimo),
        "Valor Máximo": abs(valor_final - valor_maximo),
        "Valor Médio": abs(valor_final - valor_medio),
    }
    nome_proximo = min(diffs, key=diffs.get)
    valor_proximo = {"Valor Mínimo": valor_minimo, "Valor Máximo": valor_maximo, "Valor Médio": valor_medio}[nome_proximo]

    painel_html = f"""
    <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
        {html_metric("Valor Máximo", formatar_moeda(valor_maximo))}
        {html_metric("Valor Mínimo", formatar_moeda(valor_minimo))}
        {html_metric("Valor Médio", formatar_moeda(valor_medio))}
        {html_metric("Variância", formatar_moeda(variancia))}
        {html_metric("Valor 5 anos atrás", formatar_moeda(valor_inicio), formatar_data(dt_inicio))}
        {html_metric("Valor Atual", formatar_moeda(valor_final), formatar_data(dt_final))}
    </div>
    <div style='background:orange; padding:12px; border-radius:8px; margin:16px 0;'>
        <p style='font-weight:bold; color:black; margin:0;'>
            Valor Atual está mais próximo de {nome_proximo}:
        </p>
        <p style='font-size:18px; color:black; margin:4px 0 0 0;'>
            Valor Atual: {formatar_moeda(valor_final)} | {nome_proximo}: {formatar_moeda(valor_proximo)}
        </p>
    </div>
    """
    displayHTML(painel_html)
else:
    displayHTML(html_alert("Sem dados no período de 5 anos.", "warning"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Top 3 Meses — Maiores e Menores Médias (Último Ano)

# COMMAND ----------

data_12m = datetime.now() - timedelta(days=365)
mask = (df_valores["Date"] >= data_12m) & (df_valores["Date"] <= pd.to_datetime(date.today()))
df_12m = df_valores.loc[mask].copy()

if not df_12m.empty:
    df_12m["YearMonth"] = df_12m["Date"].dt.to_period("M")
    media_mes = df_12m.groupby("YearMonth")["Close"].mean()

    top3 = media_mes.nlargest(3)
    bottom3 = media_mes.nsmallest(3)

    html_top = "<h3>🟢 Top 3 Meses (Maiores Médias — último ano)</h3><ul>"
    for ym, m in top3.items():
        html_top += f"<li><b>{ym}</b>: {formatar_moeda(m)}</li>"
    html_top += "</ul>"

    html_bottom = "<h3>🔴 Bottom 3 Meses (Menores Médias — último ano)</h3><ul>"
    for ym, m in bottom3.items():
        html_bottom += f"<li><b>{ym}</b>: {formatar_moeda(m)}</li>"
    html_bottom += "</ul>"

    displayHTML(html_top + html_bottom)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👛 Minhas Ações — Análise do Portfólio

# COMMAND ----------

if not df_tickers.empty and "ticker" in df_tickers.columns:
    # Verificar se o ticker atual está no portfólio
    ticker_busca = ticker  # Já com .SA se necessário
    acao_registro = df_tickers[df_tickers["ticker"] == ticker_busca]

    if not acao_registro.empty:
        displayHTML(html_alert(f"A ação <b>{ticker}</b> está presente nos seus ativos.", "success"))
        display(spark.createDataFrame(acao_registro))

        if "paguei" in acao_registro.columns:
            valor_pago_str = str(acao_registro.iloc[0]["paguei"]).replace(",", ".")
            try:
                valor_pago = float(valor_pago_str)
                valor_max_hist = float(df_valores["Close"].max())
                diff = valor_max_hist - valor_pago
                displayHTML(f"""
                <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
                    {html_metric("Valor Pago", formatar_moeda(valor_pago))}
                    {html_metric("Valor Máximo Histórico", formatar_moeda(valor_max_hist))}
                    {html_metric("Diferença (Max - Pago)", formatar_moeda(diff))}
                </div>
                """)
            except ValueError:
                print("Não foi possível converter o valor pago.")
    else:
        displayHTML(html_alert(f"A ação <b>{ticker}</b> não está no seu portfólio.", "info"))
else:
    displayHTML(html_alert("Tabela de tickers do portfólio não encontrada ou vazia.", "warning"))

# COMMAND ----------


