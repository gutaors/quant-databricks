# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Consolidado — Indicadores + Prophet + GPT
# MAGIC Notebook unificado com três modos de análise selecionáveis via widget.
# MAGIC
# MAGIC Equivalente a `pages/consolidado_local.py` do Streamlit.
# MAGIC
# MAGIC **Modos:**
# MAGIC - **Indicadores**: MM200, RSI, MACD com alertas e recomendações
# MAGIC - **Prophet**: Previsão com Facebook Prophet
# MAGIC - **GPT**: Previsão com GPT-2

# COMMAND ----------

# MAGIC %pip install prophet yfinance torch transformers plotly --quiet

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objs as go

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown("modo", "Indicadores", ["Indicadores", "Prophet", "GPT"], "Selecione a aplicação")
dbutils.widgets.dropdown("ticker", "", listar_tickers_disponiveis(spark), "Escolha o Ativo")
dbutils.widgets.text("data_corte", date.today().strftime("%Y-%m-%d"), "Data de Corte / Final (AAAA-MM-DD)")
dbutils.widgets.text("data_inicial", "2020-01-01", "Data Inicial (para Prophet/GPT)")
dbutils.widgets.text("meses_previsao", "6", "Meses de Previsão (1-24)")

# COMMAND ----------

modo = dbutils.widgets.get("modo")
ticker = dbutils.widgets.get("ticker")
data_corte = pd.to_datetime(dbutils.widgets.get("data_corte"))

displayHTML(f"<h2>{modo} — {ticker}</h2>")

# Carregar dados
df_all = carregar_cotacoes_ticker(spark, ticker)
if df_all.empty:
    displayHTML(html_alert(f"Nenhum dado para <b>{ticker}</b>.", "error"))
    dbutils.notebook.exit("Sem dados")

df_all = df_all.set_index("Date").sort_index()
df_all["Close"] = pd.to_numeric(df_all["Close"], errors="coerce")
df_all = df_all.dropna(subset=["Close"])

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# MODO: INDICADORES
# ═══════════════════════════════════════════════════════════════════════════════

if modo == "Indicadores":
    df_cut = df_all[df_all.index <= data_corte].copy()

    # Indicadores
    df_cut["MM200"] = df_cut["Close"].rolling(window=200).mean()
    delta = df_cut["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_cut["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df_cut["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_cut["Close"].ewm(span=26, adjust=False).mean()
    df_cut["MACD"] = ema12 - ema26
    df_cut["Signal"] = df_cut["MACD"].ewm(span=9, adjust=False).mean()

    # Gráficos
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Preço + MM200", "RSI", "MACD"),
                        vertical_spacing=0.08, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Close"], name="Close", line_color="royalblue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM200"], name="MM200", line_color="orange"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["RSI"], name="RSI", line_color="purple"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MACD"], name="MACD", line_color="cyan"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Signal"], name="Signal", line_color="magenta"), row=3, col=1)
    fig.update_layout(title=f"Indicadores — {ticker}", template="plotly_dark", height=800)
    displayHTML(fig.to_html())

    # Alertas
    if len(df_cut) >= 2:
        current = df_cut.iloc[-1]
        previous = df_cut.iloc[-2]
        tol = 0.01
        alertas = ""

        tocou_cima = previous["Close"] > previous["MM200"] and abs(current["Close"] - current["MM200"]) / current["MM200"] < tol
        tocou_baixo = previous["Close"] < previous["MM200"] and abs(current["Close"] - current["MM200"]) / current["MM200"] < tol

        if tocou_cima:
            alertas += html_alert("Preço <b>caiu para a MM200</b>.", "info")
            if current["RSI"] > 50 and current["MACD"] > current["Signal"]:
                alertas += html_alert("<b>Suporte forte</b> — Bom ponto de compra.", "success")
        if previous["Close"] > previous["MM200"] and current["Close"] < current["MM200"]:
            alertas += html_alert("<b>Rompimento baixista</b> — Considere reduzir posições.", "error")
        if tocou_baixo:
            alertas += html_alert("Preço <b>subiu para a MM200</b>.", "warning")
        if previous["Close"] < previous["MM200"] and current["Close"] > current["MM200"]:
            alertas += html_alert("<b>Rompimento altista</b> — Oportunidade de compra.", "success")

        # Recomendação
        if current["RSI"] > 50 and current["MACD"] > current["Signal"] and current["Close"] > current["MM200"]:
            alertas += html_alert("<b>Tendência Altista</b>", "success")
        elif current["RSI"] < 50 and current["MACD"] < current["Signal"] and current["Close"] < current["MM200"]:
            alertas += html_alert("<b>Tendência Baixista</b>", "error")
        else:
            alertas += html_alert("<b>Sinal Neutro/Misto</b>", "warning")

        metrics = f"""
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
            {html_metric("Preço", f"{current['Close']:.2f}")}
            {html_metric("MM200", f"{current['MM200']:.2f}")}
            {html_metric("RSI", f"{current['RSI']:.2f}")}
            {html_metric("MACD", f"{current['MACD']:.2f}")}
            {html_metric("Signal", f"{current['Signal']:.2f}")}
        </div>
        """
        displayHTML(metrics + alertas)

    # Volume
    if "Volume" in df_cut.columns:
        df_cut["Volume"] = pd.to_numeric(df_cut["Volume"], errors="coerce")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_cut.index, y=df_cut["Volume"], marker_color="rgba(100,149,237,0.5)"))
        fig_vol.update_layout(title="Volume", template="plotly_dark", height=250)
        displayHTML(fig_vol.to_html())

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# MODO: PROPHET
# ═══════════════════════════════════════════════════════════════════════════════

if modo == "Prophet":
    from prophet import Prophet
    from prophet.plot import plot_plotly

    dt_inicial = dbutils.widgets.get("data_inicial")
    meses = int(dbutils.widgets.get("meses_previsao"))

    df_p = df_all.loc[(df_all.index >= pd.to_datetime(dt_inicial)) & (df_all.index <= data_corte)].copy()

    if df_p.empty:
        displayHTML(html_alert("Sem dados no período.", "warning"))
    else:
        display(spark.createDataFrame(df_p.reset_index()))

        # Gráfico histórico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p["Close"], name="Close", line_color="royalblue"))
        fig.update_layout(title=f"Variação — {ticker}", template="plotly_dark", height=350)
        displayHTML(fig.to_html())

        # Prophet
        df_train = df_p[["Close"]].copy().reset_index()
        df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        modelo = Prophet()
        modelo.fit(df_train)
        futuro = modelo.make_future_dataframe(periods=meses * 30)
        previsoes = modelo.predict(futuro)

        fig_prev = plot_plotly(modelo, previsoes, xlabel="Período", ylabel="Valor")
        fig_prev.update_layout(template="plotly_dark", height=450)
        displayHTML(fig_prev.to_html())

        # Resumo
        ultima = previsoes.iloc[-1]
        ultimo_real = float(df_p["Close"].iloc[-1])
        displayHTML(f"""
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
            {html_metric("Último Real", formatar_moeda(ultimo_real))}
            {html_metric("Previsto", formatar_moeda(float(ultima['yhat'])))}
            {html_metric("Inferior", formatar_moeda(float(ultima['yhat_lower'])))}
            {html_metric("Superior", formatar_moeda(float(ultima['yhat_upper'])))}
        </div>
        """)

        if float(ultima["yhat"]) > ultimo_real:
            displayHTML(html_alert("<b>Tendência Altista</b> — Manter ou aumentar posições.", "success"))
        else:
            displayHTML(html_alert("<b>Tendência Baixista</b> — Cautela ou redução de posições.", "error"))

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# MODO: GPT
# ═══════════════════════════════════════════════════════════════════════════════

if modo == "GPT":
    import torch
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import datetime as dt

    end_date = data_corte.date() if hasattr(data_corte, 'date') else data_corte
    start_date = end_date - dt.timedelta(days=365)

    data = df_all[(df_all.index >= pd.to_datetime(start_date)) & (df_all.index <= pd.to_datetime(end_date))]

    if data.empty:
        displayHTML(html_alert("Sem dados no período.", "error"))
    else:
        prices = data["Close"].tolist()
        purchase_price = float(prices[-1])

        display(spark.createDataFrame(data.reset_index()))

        # GPT-2
        print("Carregando GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        max_new_tokens = 20
        max_input_length = tokenizer.model_max_length - max_new_tokens
        prompt = "Historical Prices: " + " ".join([str(p) for p in prices]) + "\nPredicted: "
        encoded = tokenizer.encode(prompt, truncation=True, max_length=max_input_length, return_tensors="pt")

        print("Gerando previsões...")
        attn = torch.ones(encoded.shape, device=encoded.device)
        gen = model.generate(encoded, attention_mask=attn, max_new_tokens=max_new_tokens,
                             temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
        predicted_text = tokenizer.decode(gen[0][encoded.shape[1]:], skip_special_tokens=True)
        print(f"Texto gerado: {predicted_text}")

        predicted_prices = []
        for tok in predicted_text.split():
            try:
                tc = re.sub(r"[^\d\.]+", "", tok)
                if tc:
                    predicted_prices.append(float(tc))
            except ValueError:
                continue

        # Gráfico
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data.index, prices, label="Histórico", color="royalblue")
        if predicted_prices:
            future_dates = data.index[-1] + pd.to_timedelta(np.arange(1, len(predicted_prices) + 1), "D")
            ax.plot(future_dates, predicted_prices, "g^", label="Previstos", markersize=10)
        ax.set_title(f"{ticker} — Histórico + GPT-2")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        display(fig)

        if predicted_prices:
            last_pred = predicted_prices[-1]
            displayHTML(f"""
            <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
                {html_metric("Último Histórico", formatar_moeda(purchase_price))}
                {html_metric("Último Previsto", formatar_moeda(last_pred))}
            </div>
            """)
            if last_pred > purchase_price:
                displayHTML(html_alert("<b>Tendência Altista</b>", "success"))
            else:
                displayHTML(html_alert("<b>Tendência Baixista</b>", "error"))
