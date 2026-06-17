# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🤖 Previsão com GPT-2
# MAGIC Previsão de preços de ações usando o modelo GPT-2 (Transformers).
# MAGIC
# MAGIC Equivalente a `pages/torch_arq_local.py` e `pages/torch.py` do Streamlit.
# MAGIC
# MAGIC > ⚠️ Nota: Este notebook requer GPU ou pode ser lento em clusters Community Edition.

# COMMAND ----------

# MAGIC %pip install torch transformers matplotlib --quiet

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datetime as dt

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown("ticker", "", listar_tickers_disponiveis(spark), "Selecione o Ticker")
dbutils.widgets.text("data_final", "2023-06-08", "Data Final (AAAA-MM-DD)")

# COMMAND ----------

# ── Carregar Dados ────────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
end_date = pd.to_datetime(dbutils.widgets.get("data_final")).date()
start_date = end_date - dt.timedelta(days=365)

df_full = carregar_cotacoes_ticker(spark, ticker)

if df_full.empty:
    displayHTML(html_alert(f"Nenhum dado para <b>{ticker}</b>.", "error"))
    dbutils.notebook.exit("Sem dados")

df_full = df_full.set_index("Date").sort_index()

displayHTML(f"""
<h2>Previsão com GPT-2 — {ticker}</h2>
<p>Período de treinamento: {start_date} a {end_date} (1 ano)</p>
""")

# COMMAND ----------

# ── Preços futuros reais (para comparação) ────────────────────────────────────

def get_price_for_date(target_date, data):
    target = pd.to_datetime(target_date)
    future = data[data.index >= target]
    if not future.empty:
        return float(future.iloc[0]["Close"])
    return None

price_7d = get_price_for_date(end_date + dt.timedelta(days=7), df_full)
price_15d = get_price_for_date(end_date + dt.timedelta(days=15), df_full)
price_30d = get_price_for_date(end_date + dt.timedelta(days=30), df_full)

# Filtrar dados históricos
data = df_full[(df_full.index >= pd.to_datetime(start_date)) & (df_full.index <= pd.to_datetime(end_date))]

if data.empty:
    displayHTML(html_alert("Nenhum dado encontrado para o intervalo selecionado.", "error"))
    dbutils.notebook.exit("Sem dados no período")

prices = data["Close"].tolist()
purchase_price = float(prices[-1])

print(f"Dados carregados: {len(data)} registros")
print(f"Preço na data final: R$ {purchase_price:.2f}")
print(f"Preço 7d após: {f'R$ {price_7d:.2f}' if price_7d else 'N/A'}")
print(f"Preço 15d após: {f'R$ {price_15d:.2f}' if price_15d else 'N/A'}")
print(f"Preço 30d após: {f'R$ {price_30d:.2f}' if price_30d else 'N/A'}")

# COMMAND ----------

# ── Dados Históricos ──────────────────────────────────────────────────────────

display(spark.createDataFrame(data.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Geração de Previsões com GPT-2

# COMMAND ----------

print("Carregando modelo GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

max_new_tokens = 20
max_input_length = tokenizer.model_max_length - max_new_tokens

prompt = "Historical Prices: " + " ".join([str(price) for price in prices]) + "\nPredicted: "
encoded_prompt = tokenizer.encode(prompt, truncation=True, max_length=max_input_length, return_tensors="pt")

print("Gerando previsões...")
attention_mask = torch.ones(encoded_prompt.shape, device=encoded_prompt.device)
generated = model.generate(
    encoded_prompt,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)

generated_tokens = generated[0]
predicted_tokens = generated_tokens[encoded_prompt.shape[1]:]
predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
print(f"Texto gerado: {predicted_text}")

# Extrair preços previstos
predicted_prices = []
for token in predicted_text.split():
    try:
        token_clean = re.sub(r"[^\d\.]+", "", token)
        if token_clean:
            predicted_prices.append(float(token_clean))
    except ValueError:
        continue

print(f"Preços previstos: {predicted_prices}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Gráfico — Preços Históricos vs Previstos

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, prices, label="Preços Históricos", color="royalblue")

if predicted_prices:
    future_dates = data.index[-1] + pd.to_timedelta(np.arange(1, len(predicted_prices) + 1), "D")
    ax.plot(future_dates, predicted_prices, "g^", label="Previstos (GPT-2)", markersize=10)

ax.set_xlabel("Data")
ax.set_ylabel("Preço (R$)")
ax.set_title(f"{ticker} — Preços Históricos e Previstos (GPT-2)")
ax.legend()
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💰 Análise de Lucro/Prejuízo

# COMMAND ----------

resultados = []
if price_7d is not None:
    profit_7d = price_7d - purchase_price
    resultados.append({"Periodo": "7 dias após", "Preco_Real": formatar_moeda(price_7d), "Lucro_Prejuizo": formatar_moeda(profit_7d)})
if price_15d is not None:
    profit_15d = price_15d - purchase_price
    resultados.append({"Periodo": "15 dias após", "Preco_Real": formatar_moeda(price_15d), "Lucro_Prejuizo": formatar_moeda(profit_15d)})
if price_30d is not None:
    profit_30d = price_30d - purchase_price
    resultados.append({"Periodo": "30 dias após", "Preco_Real": formatar_moeda(price_30d), "Lucro_Prejuizo": formatar_moeda(profit_30d)})

if resultados:
    displayHTML("<h3>Análise de Lucro/Prejuízo (Valores Reais)</h3>")
    display(spark.createDataFrame(pd.DataFrame(resultados)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Resumo e Recomendação

# COMMAND ----------

if predicted_prices:
    last_predicted = predicted_prices[-1]
    last_historical = purchase_price

    metrics_html = f"""
    <div style='display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;'>
        {html_metric("Último Valor Histórico", formatar_moeda(last_historical))}
        {html_metric("Último Valor Previsto (GPT)", formatar_moeda(last_predicted))}
    </div>
    """

    if last_predicted > last_historical:
        rec = html_alert("<b>Recomendação: Tendência Altista</b> — Recomenda-se manter ou aumentar posições de compra.", "success")
    else:
        rec = html_alert("<b>Recomendação: Tendência Baixista</b> — Recomenda-se cautela ou redução de posições.", "error")

    displayHTML(metrics_html + rec)
else:
    displayHTML(html_alert("Previsões não geradas corretamente para exibir resumo.", "error"))
