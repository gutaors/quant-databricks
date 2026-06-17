# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎲 Jogo: Alta ou Baixa?
# MAGIC Você será transportado para uma data aleatória do passado. Com base no comportamento da ação
# MAGIC nos meses anteriores, adivinhe: o preço subiu ou caiu nos próximos 2 meses?
# MAGIC
# MAGIC Equivalente a `pages/game.py` do Streamlit.
# MAGIC
# MAGIC **Como jogar:**
# MAGIC 1. Execute a célula "Sortear Cenário" para gerar uma rodada
# MAGIC 2. Analise o gráfico dos últimos 3 meses
# MAGIC 3. Defina o widget `aposta` como "alta" ou "baixa"
# MAGIC 4. Execute a célula "Revelar Resultado"

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd
import random
import plotly.graph_objs as go
from datetime import timedelta

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown("aposta", "---", ["---", "alta", "baixa"], "Sua Aposta")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎰 Sortear Cenário

# COMMAND ----------

tickers = listar_tickers_disponiveis(spark)

if not tickers:
    displayHTML(html_alert("Nenhum ticker disponível. Importe cotações primeiro.", "error"))
    dbutils.notebook.exit("Sem tickers")

# Tentar encontrar cenário válido
cenario = None
for _ in range(50):
    ticker = random.choice(tickers)
    df = carregar_cotacoes_ticker(spark, ticker)

    if len(df) < 200:
        continue

    data_min = df["Date"].min() + timedelta(days=90)
    data_max = df["Date"].max() - timedelta(days=65)

    if data_min >= data_max:
        continue

    datas_validas = df[(df["Date"] >= data_min) & (df["Date"] <= data_max)]["Date"].tolist()
    if not datas_validas:
        continue

    data_escolhida = random.choice(datas_validas)

    # Contexto: 3 meses anteriores
    df_passado = df[(df["Date"] >= data_escolhida - timedelta(days=90)) & (df["Date"] <= data_escolhida)]
    preco_atual = float(df_passado.iloc[-1]["Close"])

    # Futuro: 1 e 2 meses
    df_1m = df[df["Date"] >= data_escolhida + timedelta(days=30)]
    df_2m = df[df["Date"] >= data_escolhida + timedelta(days=60)]

    if df_1m.empty or df_2m.empty:
        continue

    cenario = {
        "ticker": ticker,
        "data_ref": data_escolhida,
        "preco_ref": preco_atual,
        "df_passado": df_passado,
        "data_1m": df_1m.iloc[0]["Date"],
        "preco_1m": float(df_1m.iloc[0]["Close"]),
        "data_2m": df_2m.iloc[0]["Date"],
        "preco_2m": float(df_2m.iloc[0]["Close"]),
    }
    break

if cenario is None:
    displayHTML(html_alert("Não foi possível gerar cenário. Tente novamente.", "error"))
    dbutils.notebook.exit("Sem cenário")

# COMMAND ----------

# ── Exibir Cenário ────────────────────────────────────────────────────────────

displayHTML(f"""
<div style='background:#1e1e1e; padding:20px; border-radius:10px; margin:16px 0;'>
    <h2 style='color:white;'>🎲 Ação Sorteada: {cenario['ticker']}</h2>
    <p style='color:#aaa; font-size:16px;'>
        📅 <b>Data de Referência:</b> {formatar_data(cenario['data_ref'])}<br/>
        💵 <b>Preço na época:</b> {formatar_moeda(cenario['preco_ref'])}
    </p>
</div>
""")

# COMMAND ----------

# ── Gráfico de Contexto (3 meses anteriores) ─────────────────────────────────

displayHTML("<h3>📈 Contexto Histórico (Últimos 3 meses antes da data sorteada)</h3>")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cenario["df_passado"]["Date"],
    y=cenario["df_passado"]["Close"],
    name="Preço de Fechamento",
    line=dict(color="royalblue", width=2),
    fill="tozeroy",
    fillcolor="rgba(65,105,225,0.1)"
))
fig.update_layout(
    title=f"Contexto — {cenario['ticker']} (3 meses antes de {formatar_data(cenario['data_ref'])})",
    template="plotly_dark", height=350,
    xaxis_title="Data", yaxis_title="Preço (R$)"
)
displayHTML(fig.to_html())

displayHTML("""
<div style='background:#263238; padding:12px; border-radius:6px; margin:8px 0; color:#b0bec5;'>
    <b>💡 Dica:</b> Analise a tendência do gráfico acima e defina o widget <b>"Sua Aposta"</b>
    como <b>"alta"</b> ou <b>"baixa"</b>, depois execute a célula "Revelar Resultado".
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Revelar Resultado

# COMMAND ----------

aposta = dbutils.widgets.get("aposta")

if aposta == "---":
    displayHTML(html_alert("⬆️ Defina o widget <b>Sua Aposta</b> como 'alta' ou 'baixa' e re-execute esta célula.", "info"))
else:
    preco_ref = cenario["preco_ref"]
    preco_2m = cenario["preco_2m"]
    houve_alta = preco_2m > preco_ref
    acertou = (aposta == "alta" and houve_alta) or (aposta == "baixa" and not houve_alta)

    var_1m = ((cenario["preco_1m"] - preco_ref) / preco_ref) * 100
    var_2m = ((preco_2m - preco_ref) / preco_ref) * 100

    if acertou:
        resultado_html = f"""
        <div style='background:#0f9d58; color:white; padding:24px; border-radius:10px; text-align:center; margin:16px 0;'>
            <h1 style='color:white; margin:0;'>🎉 PARABÉNS! Você acertou!</h1>
            <p style='color:white; font-size:18px;'>Sua aposta: <b>{'ALTA 📈' if aposta == 'alta' else 'BAIXA 📉'}</b></p>
        </div>
        """
    else:
        resultado_html = f"""
        <div style='background:#dc3545; color:white; padding:24px; border-radius:10px; text-align:center; margin:16px 0;'>
            <h1 style='color:white; margin:0;'>❌ QUE PENA! Você errou!</h1>
            <p style='color:white; font-size:18px;'>Sua aposta: <b>{'ALTA 📈' if aposta == 'alta' else 'BAIXA 📉'}</b></p>
        </div>
        """

    # Tabela de resultados
    dados_tabela = pd.DataFrame([
        {"Periodo": "Data Inicial", "Data": formatar_data(cenario["data_ref"]),
         "Preco": formatar_moeda(preco_ref), "Variacao": "-"},
        {"Periodo": "Após 1 Mês", "Data": formatar_data(cenario["data_1m"]),
         "Preco": formatar_moeda(cenario["preco_1m"]), "Variacao": f"{var_1m:+.2f}%"},
        {"Periodo": "Após 2 Meses", "Data": formatar_data(cenario["data_2m"]),
         "Preco": formatar_moeda(preco_2m), "Variacao": f"{var_2m:+.2f}%"},
    ])

    displayHTML(resultado_html)
    display(spark.createDataFrame(dados_tabela))

    displayHTML("<p><i>Para jogar novamente, re-execute a célula 'Sortear Cenário'.</i></p>")
