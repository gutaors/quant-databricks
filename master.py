# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🌟 Master Quant
# MAGIC Análise completa de um ativo consolidando todos os módulos do projeto.

# COMMAND ----------

# MAGIC %pip install yfinance prophet torch transformers plotly beautifulsoup4 --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import urllib.request
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from prophet import Prophet
from prophet.plot import plot_plotly
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

# COMMAND ----------

# ── Widgets e Configuração Única ──────────────────────────────────────────────

dbutils.widgets.removeAll()
dbutils.widgets.text("ticker", "PETR4.SA", "Ticker (Ex: PETR4.SA)")
dbutils.widgets.text("data_inicial", "2020-01-01", "Data Inicial (AAAA-MM-DD)")
dbutils.widgets.text("data_corte", date.today().strftime("%Y-%m-%d"), "Data Final/Corte (AAAA-MM-DD)")
dbutils.widgets.text("meses_previsao", "6", "Meses de Previsão Prophet")

# COMMAND ----------

ticker_input = dbutils.widgets.get("ticker").strip().upper()
if not ticker_input.endswith(".SA") and not ticker_input.endswith(".US") and not ticker_input.endswith(".sa"):
    ticker_input += ".SA"
ticker = ticker_input

data_inicial = pd.to_datetime(dbutils.widgets.get("data_inicial"))
data_corte = pd.to_datetime(dbutils.widgets.get("data_corte"))
meses_previsao = int(dbutils.widgets.get("meses_previsao"))

displayHTML(f"<h1>🚀 Análise Master para {ticker}</h1>")
displayHTML(f"<p><b>Período analisado:</b> {data_inicial.date()} a {data_corte.date()}</p>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importação de Cotações (Módulos 03, 01, 04)

# COMMAND ----------

print(f"[{ticker}] - Importando cotações entre {data_inicial.date()} e {data_corte.date()}...")

# Tenta baixar o período de análise via yfinance
df_all = pd.DataFrame()
try:
    df_all = yf.download(ticker, start=data_inicial.strftime("%Y-%m-%d"), end=(data_corte + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, multi_level_index=False).reset_index()
except TypeError:
    df_all = yf.download(ticker, start=data_inicial.strftime("%Y-%m-%d"), end=(data_corte + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False).reset_index()

if df_all.empty:
    displayHTML(html_alert(f"Nenhum dado encontrado para {ticker} no período.", "error"))
    dbutils.notebook.exit("Falha na importação")

df_all["Ticker"] = ticker
try:
    salvar_cotacoes_delta(spark, df_all, ticker, "append") # Salva na tabela Delta localmente para manter o histórico
except Exception as e:
    print(f"Aviso: Não foi possível salvar no Delta Lake: {e}")

df_all = df_all.set_index("Date").sort_index()
df_all["Close"] = pd.to_numeric(df_all["Close"], errors="coerce")
df_all = df_all.dropna(subset=["Close"])

displayHTML(html_alert(f"✅ Dados de {ticker} atualizados e carregados! Registros: {len(df_all)}", "success"))

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df_all.index, y=df_all["Close"], name="Preço", line_color="royalblue"))
fig_hist.update_layout(title=f"Histórico de Preços — {ticker}", template="plotly_dark", height=400)
displayHTML(fig_hist.to_html())

# Resumo Básico
min_price = df_all["Close"].min()
max_price = df_all["Close"].max()
last_price = df_all["Close"].iloc[-1]
displayHTML(f"""
<div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
    {html_metric("Último Preço", formatar_moeda(last_price))}
    {html_metric("Mínimo do Período", formatar_moeda(min_price))}
    {html_metric("Máximo do Período", formatar_moeda(max_price))}
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Indicadores Técnicos (Módulos 05 e 06)

# COMMAND ----------

print(f"[{ticker}] - Calculando indicadores técnicos (MM50, MM200, RSI, MACD)...")

df_cut = df_all.copy()
df_cut["MM50"] = df_cut["Close"].rolling(window=50).mean()
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

fig_ind = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("Preço + MM50 + MM200", "RSI", "MACD"),
                    vertical_spacing=0.08, row_heights=[0.5, 0.25, 0.25])

fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Close"], name="Close", line_color="royalblue"), row=1, col=1)
fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM50"], name="MM50", line_color="yellow"), row=1, col=1)
fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MM200"], name="MM200", line_color="orange"), row=1, col=1)

fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["RSI"], name="RSI", line_color="purple"), row=2, col=1)
fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["MACD"], name="MACD", line_color="cyan"), row=3, col=1)
fig_ind.add_trace(go.Scatter(x=df_cut.index, y=df_cut["Signal"], name="Signal", line_color="magenta"), row=3, col=1)

fig_ind.update_layout(title=f"Indicadores Técnicos — {ticker}", template="plotly_dark", height=800)
displayHTML(fig_ind.to_html())

if len(df_cut) >= 2:
    current = df_cut.iloc[-1]
    alertas = "<h3>Análise Automática:</h3>"
    
    if current["RSI"] > 70:
        alertas += html_alert("RSI indica possível SOBRECOMPRA (>70).", "warning")
    elif current["RSI"] < 30:
        alertas += html_alert("RSI indica possível SOBREVENDA (<30, favorável à compra).", "success")
    else:
        alertas += html_alert("RSI indica momento NEUTRO.", "info")
    
    if current["Close"] > current["MM200"]:
        alertas += html_alert("Tendência Primária de ALTA (Preço > MM200).", "success")
    else:
        alertas += html_alert("Tendência Primária de BAIXA (Preço < MM200).", "error")
        
    displayHTML(alertas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Variação de 10% (Módulo 09)

# COMMAND ----------

print(f"[{ticker}] - Calculando variações extremas...")

primeiro_preco = df_all["Close"].iloc[0]
ultimo_preco = df_all["Close"].iloc[-1]
variacao = ((ultimo_preco - primeiro_preco) / primeiro_preco) * 100

cor_var = "success" if variacao >= 10 else ("error" if variacao <= -10 else "info")
if abs(variacao) >= 10:
    displayHTML(html_alert(f"O ativo {ticker} teve variação de <b>{variacao:.2f}%</b> no período ({formatar_moeda(primeiro_preco)} ➔ {formatar_moeda(ultimo_preco)}).", cor_var))
else:
    displayHTML(html_alert(f"O ativo {ticker} variou menos de 10% no período total: <b>{variacao:.2f}%</b>.", cor_var))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Previsão com Prophet (Módulo 07)

# COMMAND ----------

print(f"[{ticker}] - Gerando previsão Prophet para os próximos {meses_previsao} meses...")

df_train = df_all[["Close"]].copy().reset_index()
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
modelo = Prophet()
modelo.fit(df_train)
futuro = modelo.make_future_dataframe(periods=meses_previsao * 30)
previsoes = modelo.predict(futuro)

fig_prev = plot_plotly(modelo, previsoes, xlabel="Período", ylabel="Valor")
fig_prev.update_layout(template="plotly_dark", height=450, title=f"Previsão Prophet — {ticker}")
displayHTML(fig_prev.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Previsão com GPT-2 (Módulo 08)

# COMMAND ----------

print(f"[{ticker}] - Gerando previsão preditiva usando GPT-2...")

prices = df_all["Close"].tolist()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

max_new_tokens = 20
max_input_length = tokenizer.model_max_length - max_new_tokens
prompt = "Historical Prices: " + " ".join([str(p) for p in prices[-100:]]) + "\nPredicted: "
encoded = tokenizer.encode(prompt, truncation=True, max_length=max_input_length, return_tensors="pt")

attn = torch.ones(encoded.shape, device=encoded.device)
gen = model.generate(encoded, attention_mask=attn, max_new_tokens=max_new_tokens,
                     temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
predicted_text = tokenizer.decode(gen[0][encoded.shape[1]:], skip_special_tokens=True)

predicted_prices = []
for tok in predicted_text.split():
    try:
        tc = re.sub(r"[^\d\.]+", "", tok)
        if tc:
            predicted_prices.append(float(tc))
    except ValueError:
        continue

fig_gpt, ax_gpt = plt.subplots(figsize=(14, 6))
ax_gpt.plot(df_all.index, prices, label="Histórico", color="royalblue")
if predicted_prices:
    future_dates = df_all.index[-1] + pd.to_timedelta(np.arange(1, len(predicted_prices) + 1), "D")
    ax_gpt.plot(future_dates, predicted_prices, "g^", label="Previstos", markersize=10)
ax_gpt.set_title(f"{ticker} — Histórico + GPT-2")
ax_gpt.legend()
ax_gpt.tick_params(axis="x", rotation=45)
fig_gpt.tight_layout()
display(fig_gpt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Recomendações de Bancos e Notícias (Módulos 10 e 11)

# COMMAND ----------

print(f"[{ticker}] - Buscando recomendações de analistas (Yahoo Finance)...")

try:
    ativo_yf = yf.Ticker(ticker)
    df_recomendacoes = ativo_yf.get_recommendations()

    if df_recomendacoes is not None and not df_recomendacoes.empty:
        displayHTML(html_alert(f"Foram encontradas avaliações recentes para {ticker}.", "success"))
        colunas_voto = [c for c in df_recomendacoes.columns if c.lower() in ['strongbuy', 'buy', 'hold', 'sell', 'strongsell']]
        if colunas_voto:
            fig_rec = go.Figure()
            cores = {'strongbuy': '#0f9d58', 'buy': '#4caf50', 'hold': '#ff9800', 'sell': '#f44336', 'strongsell': '#b71c1c'}
            for col in colunas_voto:
                fig_rec.add_trace(go.Bar(
                    name=col.capitalize(),
                    x=df_recomendacoes.index.astype(str),
                    y=df_recomendacoes[col],
                    marker_color=cores.get(col.lower(), '#666')
                ))
            fig_rec.update_layout(barmode='group', title=f"Recomendações Bancos — {ticker}", template="plotly_dark", height=400)
            displayHTML(fig_rec.to_html())
    else:
        displayHTML(html_alert(f"Sem recomendações no Yahoo Finance para {ticker}.", "warning"))
except Exception as e:
    displayHTML(html_alert(f"Erro nas recomendações: {e}", "warning"))

# COMMAND ----------

print(f"[{ticker}] - Buscando notícias recentes (Google News)...")

def get_company_name(t):
    nomes = {"PETR4.SA": "Petrobras", "VALE3.SA": "Vale", "ITUB4.SA": "Itaú Unibanco"}
    base = t.replace(".SA", "")
    return nomes.get(t, base)

company_name = get_company_name(ticker)
query = urllib.parse.quote(f"{company_name} OR {ticker.replace('.SA','')} ação bolsa")
url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req, timeout=10)
    root = ET.fromstring(response.read())
    
    html_articles = "<h3>📰 Últimas Notícias</h3>"
    count = 0
    for item in root.findall(".//item")[:5]:
        title = item.find("title").text
        link = item.find("link").text
        pub = item.find("pubDate").text
        html_articles += f"""
        <div style='background:#1e1e1e; padding:10px; border-radius:5px; margin:8px 0;'>
            <p style='color:white; margin:2px 0;'><b>{title}</b></p>
            <p style='color:#888; font-size:12px; margin:2px 0;'>{pub}</p>
            <a href='{link}' target='_blank' style='color:#4fc3f7; font-size:12px;'>Ler notícia</a>
        </div>
        """
        count += 1
    if count > 0:
        displayHTML(html_articles)
    else:
        displayHTML(html_alert("Nenhuma notícia encontrada.", "warning"))
except Exception as e:
    displayHTML(html_alert(f"Erro ao buscar notícias: {e}", "warning"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusão Consolidada (Módulo 14)

# COMMAND ----------

displayHTML("<h2>🎯 Resumo e Finalização</h2>")
displayHTML(f"<p>Análise completa do ativo <b>{ticker}</b> executada e consolidada com sucesso.</p>")
displayHTML(f"<p><b>Data analisada:</b> de {data_inicial.date()} até {data_corte.date()}</p>")
displayHTML(html_alert("<b>Aviso:</b> Lembre-se de avaliar conjuntamente os indicadores técnicos, as previsões (Prophet/GPT) e as notícias fundamentais antes de qualquer decisão de investimento. O conteúdo não representa recomendação de compra direta.", "info"))
