# Databricks notebook source
# MAGIC %md
# MAGIC # 📦 Simulador em Lote — Quanto valeria hoje?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## O que este notebook faz?
# MAGIC
# MAGIC Dado um **valor de investimento**, uma **data de entrada** e uma **lista de tickers**,
# MAGIC este notebook responde a pergunta:
# MAGIC
# MAGIC > *"Se eu tivesse investido **R$ X** em cada uma dessas ações nessa data,
# MAGIC >  quanto teria após **1 mês** e após **2 meses**?"*
# MAGIC
# MAGIC ### Fluxo de execução:
# MAGIC
# MAGIC ```
# MAGIC ① Defina as variáveis (célula abaixo)
# MAGIC      ↓
# MAGIC ② Download de preços históricos (Yahoo Finance)
# MAGIC      ↓
# MAGIC ③ Calcula cotas compradas na data de entrada
# MAGIC      ↓
# MAGIC ④ Busca preços em T+1 mês e T+2 meses
# MAGIC      ↓
# MAGIC ⑤ Exibe resultado visual: ganho / perda por ativo
# MAGIC      ↓
# MAGIC ⑥ Resumo consolidado do portfólio
# MAGIC      ↓
# MAGIC ⑦ Gráfico de evolução do patrimônio
# MAGIC ```
# MAGIC
# MAGIC > 💡 **Nota:** Usamos preços de **fechamento ajustado** (auto_adjust=True),
# MAGIC > que já incorporam dividendos e desdobramentos.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ⚙️ CÉLULA DE CONFIGURAÇÃO — edite aqui!
# MAGIC
# MAGIC Esta é a **única célula que você precisa modificar**.
# MAGIC Preencha os três parâmetros abaixo e execute o notebook inteiro.

# COMMAND ----------

# DBTITLE 1,▶ CONFIGURAÇÃO — Edite esta célula
# =============================================================
#  PARÂMETROS DE SIMULAÇÃO
#  Edite os três blocos abaixo e execute o notebook.
# =============================================================

# ── 1) DATA DE ENTRADA ────────────────────────────────────────
#    Formato: "YYYY-MM-DD"
#    A data deve ser um pregão válido (dia útil).
#    Se for fim de semana ou feriado, o notebook usa o próximo pregão disponível.
DATA_ENTRADA = "2024-01-02"

# ── 2) VALOR INVESTIDO POR AÇÃO (em R$) ───────────────────────
#    Este valor será investido INDIVIDUALMENTE em CADA ticker.
#    Exemplo: R$ 1.000 em PETR4 + R$ 1.000 em VALE3 + ...
VALOR_POR_ACAO = 1000.00

# ── 3) LISTA DE TICKERS ───────────────────────────────────────
#    Use o código de negociação da B3 + ".SA" (padrão Yahoo Finance)
#    Exemplos: "PETR4.SA", "VALE3.SA", "WEGE3.SA"
TICKERS = [
    "PETR4.SA",
    "VALE3.SA",
    "WEGE3.SA",
    "ITUB4.SA",
    "BBAS3.SA",
    "ABEV3.SA",
    "RENT3.SA",
    "RADL3.SA",
    "EGIE3.SA",
    "PRIO3.SA",
]

# =============================================================
#  FIM DA CONFIGURAÇÃO — não precisa editar abaixo desta linha
# =============================================================
print("✅ Configuração carregada com sucesso!")
print(f"   📅 Data de entrada   : {DATA_ENTRADA}")
print(f"   💰 Valor por ação    : R$ {VALOR_POR_ACAO:,.2f}")
print(f"   📋 Tickers           : {len(TICKERS)} ativos")
print()
for tk in TICKERS:
    print(f"      • {tk}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📦 Instalação e Importações
# MAGIC
# MAGIC O notebook usa as seguintes bibliotecas:
# MAGIC - **yfinance**: busca dados históricos de preços no Yahoo Finance (gratuito)
# MAGIC - **pandas**: manipulação de tabelas de dados
# MAGIC - **numpy**: cálculos numéricos
# MAGIC - **matplotlib**: geração de gráficos
# MAGIC - **base64 / BytesIO**: para embutir os gráficos diretamente no HTML do Databricks

# COMMAND ----------

# DBTITLE 1,Instalação de dependências
%pip install yfinance --quiet

# COMMAND ----------

# DBTITLE 1,Importações
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import base64
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")          # backend sem janela — compatível com Databricks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from IPython.display import display, HTML

warnings.filterwarnings("ignore")

print("✅ Bibliotecas importadas!")
print(f"   yfinance  : {yf.__version__}")
print(f"   pandas    : {pd.__version__}")
print(f"   numpy     : {np.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📅 Validação e Cálculo das Datas
# MAGIC
# MAGIC A partir da data de entrada, calculamos automaticamente:
# MAGIC - **T+0**: data de entrada (compra)
# MAGIC - **T+1M**: aproximadamente 21 pregões depois (~1 mês útil)
# MAGIC - **T+2M**: aproximadamente 42 pregões depois (~2 meses úteis)
# MAGIC
# MAGIC O período de download é de 3 meses a partir da data de entrada
# MAGIC para garantir que temos dados suficientes.

# COMMAND ----------

# DBTITLE 1,Validação das datas e parâmetros
# ── Converte a data de entrada ──────────────────────────────────────────────
try:
    DT_ENTRADA = datetime.strptime(DATA_ENTRADA, "%Y-%m-%d")
except ValueError:
    raise ValueError(f"❌ Formato de data inválido: '{DATA_ENTRADA}'. Use YYYY-MM-DD.")

HOJE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# A data não pode ser futura
if DT_ENTRADA >= HOJE:
    raise ValueError(
        f"❌ A data de entrada ({DATA_ENTRADA}) deve ser no passado. "
        f"Data atual: {HOJE.strftime('%Y-%m-%d')}"
    )

# ── Calcula os marcos de tempo ──────────────────────────────────────────────
# Usamos dias calendário como proxy; o notebook vai encontrar
# o pregão mais próximo disponível automaticamente
DT_1M  = DT_ENTRADA + timedelta(days=30)   # ~1 mês
DT_2M  = DT_ENTRADA + timedelta(days=60)   # ~2 meses
DT_FIM = min(DT_ENTRADA + timedelta(days=95), HOJE)  # janela de download

# Verifica se temos dados suficientes
dias_disponíveis = (HOJE - DT_ENTRADA).days
if dias_disponíveis < 30:
    print("⚠️  ATENÇÃO: Menos de 30 dias desde a entrada.")
    print("   Só será possível calcular o resultado parcial (menos de 1 mês).")

# ── Validação dos tickers ───────────────────────────────────────────────────
if not TICKERS:
    raise ValueError("❌ A lista de TICKERS está vazia!")

if VALOR_POR_ACAO <= 0:
    raise ValueError(f"❌ VALOR_POR_ACAO deve ser positivo. Recebido: {VALOR_POR_ACAO}")

# ── Resumo ──────────────────────────────────────────────────────────────────
INVESTIMENTO_TOTAL = VALOR_POR_ACAO * len(TICKERS)

print("=" * 60)
print("  PARÂMETROS DA SIMULAÇÃO")
print("=" * 60)
print(f"  Data de entrada  (T+0) : {DT_ENTRADA.strftime('%d/%m/%Y')} ({DT_ENTRADA.strftime('%A')})")
print(f"  Marco 1 mês     (T+1M) : {DT_1M.strftime('%d/%m/%Y')}")
print(f"  Marco 2 meses   (T+2M) : {DT_2M.strftime('%d/%m/%Y')}")
print(f"  Dias disponíveis       : {dias_disponíveis}")
print(f"  Ativos na carteira     : {len(TICKERS)}")
print(f"  Valor por ativo        : R$ {VALOR_POR_ACAO:,.2f}")
print(f"  Investimento total     : R$ {INVESTIMENTO_TOTAL:,.2f}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🔽 Download dos Dados Históricos
# MAGIC
# MAGIC O notebook baixa os preços diários (abertura, máxima, mínima, **fechamento ajustado**, volume)
# MAGIC de cada ativo no período da simulação.
# MAGIC
# MAGIC O **fechamento ajustado** já incorpora eventos corporativos como dividendos, bonificações
# MAGIC e desdobramentos — é o preço correto para cálculo de retorno real.

# COMMAND ----------

# DBTITLE 1,Download dos dados de preços
print(f"🔽 Baixando preços de {len(TICKERS)} ativos...")
print(f"   Período: {DT_ENTRADA.strftime('%d/%m/%Y')} → {DT_FIM.strftime('%d/%m/%Y')}\n")

# Baixa todos os tickers de uma vez (mais eficiente)
try:
    raw = yf.download(
        TICKERS,
        start=DT_ENTRADA.strftime("%Y-%m-%d"),
        end=(DT_FIM + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,   # preços ajustados por dividendos e splits
        progress=False,
        threads=True,
    )
except Exception as e:
    raise RuntimeError(f"❌ Falha no download: {e}")

# ── Organiza em um dicionário {ticker: DataFrame} ───────────────────────────
PRECOS = {}

if isinstance(raw.columns, pd.MultiIndex):
    # Múltiplos tickers → MultiIndex
    for tk in TICKERS:
        try:
            df = raw.xs(tk, axis=1, level=1).copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) >= 2:
                PRECOS[tk] = df
        except Exception:
            pass
else:
    # Apenas 1 ticker
    raw.dropna(subset=["Close"], inplace=True)
    if len(raw) >= 2:
        PRECOS[TICKERS[0]] = raw

# ── Relatório de disponibilidade ────────────────────────────────────────────
SEM_DADOS = [tk for tk in TICKERS if tk not in PRECOS]

print(f"✅ Download concluído!")
print(f"   Ativos com dados    : {len(PRECOS)}")
if SEM_DADOS:
    print(f"   ⚠️  Sem dados        : {', '.join(SEM_DADOS)}")
    print("      (verifique se o ticker está correto e se havia negociação nessa data)")
print()

# Exibe amostra dos dados
if PRECOS:
    primeiro = list(PRECOS.keys())[0]
    print(f"📊 Amostra — {primeiro}:")
    display(PRECOS[primeiro].tail(3).round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🔢 Como funciona o cálculo?
# MAGIC
# MAGIC Para cada ativo, o cálculo segue 4 passos:
# MAGIC
# MAGIC **Passo 1 — Preço de entrada (T+0)**
# MAGIC > O preço de fechamento do pregão mais próximo da data informada.
# MAGIC > Se a data for fim de semana ou feriado, usa-se o próximo pregão disponível.
# MAGIC
# MAGIC **Passo 2 — Quantidade de cotas compradas**
# MAGIC > ```
# MAGIC > cotas = VALOR_POR_ACAO / preço_entrada
# MAGIC > ```
# MAGIC > Na prática, ações só podem ser compradas em números inteiros.
# MAGIC > Este notebook usa cotas fracionárias para fins de simulação matemática.
# MAGIC
# MAGIC **Passo 3 — Preço nos marcos T+1M e T+2M**
# MAGIC > O fechamento do pregão mais próximo de cada marco temporal.
# MAGIC > Se o marco ainda não ocorreu (data futura), exibe "Aguardando".
# MAGIC
# MAGIC **Passo 4 — Valor do portfólio**
# MAGIC > ```
# MAGIC > valor_1m = cotas × preço_1m
# MAGIC > valor_2m = cotas × preço_2m
# MAGIC > retorno  = (valor_final - valor_inicial) / valor_inicial × 100
# MAGIC > ```

# COMMAND ----------

# DBTITLE 1,Função auxiliar — encontra o preço mais próximo de uma data
def preco_mais_proximo(df_ticker: pd.DataFrame, data_alvo: datetime):
    """
    Retorna (data_real, preco_fechamento) do pregão mais próximo da data_alvo.

    Busca primeiro na data exata; se não encontrar,
    procura nos 5 pregões seguintes e nos 5 anteriores.

    Parâmetros:
        df_ticker : DataFrame com coluna 'Close' e índice DatetimeIndex
        data_alvo : datetime da data desejada

    Retorna:
        (data_real: datetime, preco: float) ou (None, None) se não encontrado
    """
    if df_ticker is None or len(df_ticker) == 0:
        return None, None

    idx = df_ticker.index

    # Tenta a data exata
    ts = pd.Timestamp(data_alvo)
    if ts in idx:
        return ts.to_pydatetime(), float(df_ticker.loc[ts, "Close"])

    # Procura o pregão mais próximo APÓS a data (futuro próximo)
    futuros = idx[idx >= ts]
    if len(futuros) > 0:
        dt_real = futuros[0]
        return dt_real.to_pydatetime(), float(df_ticker.loc[dt_real, "Close"])

    # Procura o pregão mais próximo ANTES da data (passado próximo)
    passados = idx[idx < ts]
    if len(passados) > 0:
        dt_real = passados[-1]
        return dt_real.to_pydatetime(), float(df_ticker.loc[dt_real, "Close"])

    return None, None


print("✅ Função de busca de preço definida.")
print()
print("Exemplo de uso:")
if PRECOS:
    ex_tk = list(PRECOS.keys())[0]
    ex_dt, ex_pc = preco_mais_proximo(PRECOS[ex_tk], DT_ENTRADA)
    if ex_dt:
        print(f"   {ex_tk} em {ex_dt.strftime('%d/%m/%Y')} → R$ {ex_pc:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 💰 Cálculo do Portfólio — Retornos em 1 e 2 Meses
# MAGIC
# MAGIC Agora aplicamos o cálculo a todos os ativos da lista.
# MAGIC Cada linha da tabela abaixo representa um ativo.

# COMMAND ----------

# DBTITLE 1,Cálculo dos retornos para cada ativo
print("🔢 Calculando retornos...\n")

resultados = []

for ticker in TICKERS:
    # ── Verifica disponibilidade de dados ───────────────────────────────────
    if ticker not in PRECOS:
        resultados.append({
            "ticker":        ticker.replace(".SA", ""),
            "ticker_raw":    ticker,
            "data_entrada":  "—",
            "preco_entrada": None,
            "cotas":         None,
            "data_1m":       "—",
            "preco_1m":      None,
            "valor_1m":      None,
            "retorno_1m":    None,
            "data_2m":       "—",
            "preco_2m":      None,
            "valor_2m":      None,
            "retorno_2m":    None,
            "status":        "❌ Sem dados",
        })
        continue

    df = PRECOS[ticker]

    # ── Passo 1: Preço de entrada ────────────────────────────────────────────
    dt_ent, pc_ent = preco_mais_proximo(df, DT_ENTRADA)
    if pc_ent is None or pc_ent <= 0:
        resultados.append({
            "ticker":     ticker.replace(".SA",""),
            "ticker_raw": ticker,
            "status":     "❌ Sem preço de entrada",
        })
        continue

    # ── Passo 2: Quantidade de cotas ─────────────────────────────────────────
    cotas = VALOR_POR_ACAO / pc_ent

    # ── Passo 3: Preço em T+1M ───────────────────────────────────────────────
    dt_1m, pc_1m = preco_mais_proximo(df, DT_1M)
    if pc_1m is not None:
        val_1m = cotas * pc_1m
        ret_1m = (val_1m - VALOR_POR_ACAO) / VALOR_POR_ACAO * 100
    else:
        val_1m = ret_1m = None  # data futura ou sem dados

    # ── Passo 4: Preço em T+2M ───────────────────────────────────────────────
    dt_2m, pc_2m = preco_mais_proximo(df, DT_2M)
    if pc_2m is not None:
        val_2m = cotas * pc_2m
        ret_2m = (val_2m - VALOR_POR_ACAO) / VALOR_POR_ACAO * 100
    else:
        val_2m = ret_2m = None

    # ── Monta o resultado ────────────────────────────────────────────────────
    resultados.append({
        "ticker":        ticker.replace(".SA", ""),
        "ticker_raw":    ticker,
        "data_entrada":  dt_ent.strftime("%d/%m/%Y") if dt_ent else "—",
        "preco_entrada": round(pc_ent, 2),
        "cotas":         round(cotas, 4),
        "data_1m":       dt_1m.strftime("%d/%m/%Y") if dt_1m else "Aguardando",
        "preco_1m":      round(pc_1m, 2) if pc_1m else None,
        "valor_1m":      round(val_1m, 2) if val_1m else None,
        "retorno_1m":    round(ret_1m, 2) if ret_1m is not None else None,
        "data_2m":       dt_2m.strftime("%d/%m/%Y") if dt_2m else "Aguardando",
        "preco_2m":      round(pc_2m, 2) if pc_2m else None,
        "valor_2m":      round(val_2m, 2) if val_2m else None,
        "retorno_2m":    round(ret_2m, 2) if ret_2m is not None else None,
        "status":        "✅ OK",
    })

    print(f"  {ticker.replace('.SA',''):8s} | "
          f"Entrada: R$ {pc_ent:8.2f} | "
          f"1M: {'R$ {:8.2f}'.format(val_1m) if val_1m else 'Aguardando':>15s} "
          f"({'%+.1f%%'.format(ret_1m) if ret_1m is not None else '-':>8s}) | "
          f"2M: {'R$ {:8.2f}'.format(val_2m) if val_2m else 'Aguardando':>15s} "
          f"({'%+.1f%%'.format(ret_2m) if ret_2m is not None else '-':>8s})")

print()
print(f"✅ Cálculo concluído para {sum(1 for r in resultados if r['status']=='✅ OK')} ativos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📊 Tabela Detalhada por Ativo

# COMMAND ----------

# DBTITLE 1,Tabela pandas de resultados
df_res = pd.DataFrame(resultados)
cols_display = [
    "ticker", "data_entrada", "preco_entrada", "cotas",
    "data_1m", "preco_1m", "valor_1m", "retorno_1m",
    "data_2m", "preco_2m", "valor_2m", "retorno_2m",
    "status"
]
cols_ok = [c for c in cols_display if c in df_res.columns]
df_show = df_res[cols_ok].rename(columns={
    "ticker":        "Ativo",
    "data_entrada":  "Data Entrada",
    "preco_entrada": "Preço Entrada (R$)",
    "cotas":         "Cotas",
    "data_1m":       "Data T+1M",
    "preco_1m":      "Preço 1M (R$)",
    "valor_1m":      "Valor 1M (R$)",
    "retorno_1m":    "Retorno 1M (%)",
    "data_2m":       "Data T+2M",
    "preco_2m":      "Preço 2M (R$)",
    "valor_2m":      "Valor 2M (R$)",
    "retorno_2m":    "Retorno 2M (%)",
    "status":        "Status",
})
display(df_show)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🏆 Painel Visual — Resultado por Ativo
# MAGIC
# MAGIC Cada card mostra o resultado de um ativo:
# MAGIC - 🟢 Verde = lucro | 🔴 Vermelho = prejuízo | 🟡 Amarelo = sem dados ainda

# COMMAND ----------

# DBTITLE 1,Painel HTML visual por ativo
def formatar_retorno_badge(retorno, valor):
    """Gera um badge colorido para o retorno."""
    if retorno is None:
        return ('<span style="background:#45475A;color:#888;padding:3px 10px;'
                'border-radius:10px;font-size:12px;">Aguardando</span>')
    cor  = "#00C851" if retorno >= 0 else "#FF4444"
    seta = "▲" if retorno >= 0 else "▼"
    return (f'<span style="background:{cor};color:#000;padding:3px 10px;'
            f'border-radius:10px;font-size:12px;font-weight:700;">'
            f'{seta} {retorno:+.2f}%</span>'
            f'<br><small style="color:#CDD6F4;font-size:11px;">'
            f'R$ {valor:,.2f}</small>' if valor is not None else "")


def gerar_painel_html(resultados, valor_por, dt_entrada_str, dt_1m_str, dt_2m_str):
    total_inv    = valor_por * len([r for r in resultados if r["status"] == "✅ OK"])
    total_val_1m = sum(r["valor_1m"] for r in resultados if r.get("valor_1m") is not None)
    total_val_2m = sum(r["valor_2m"] for r in resultados if r.get("valor_2m") is not None)
    lucro_1m     = total_val_1m - total_inv if total_val_1m else None
    lucro_2m     = total_val_2m - total_inv if total_val_2m else None
    ret_port_1m  = (lucro_1m / total_inv * 100) if lucro_1m is not None and total_inv > 0 else None
    ret_port_2m  = (lucro_2m / total_inv * 100) if lucro_2m is not None and total_inv > 0 else None

    # Cabeçalho do portfólio
    def card_resumo(label, val, ret, dt):
        if val is None:
            return f"""<div style="background:#313244;border-radius:12px;padding:14px 20px;flex:1;min-width:180px;">
              <div style="color:#888;font-size:12px;">{label}</div>
              <div style="color:#888;font-size:22px;font-weight:800;">Aguardando</div>
              <div style="color:#888;font-size:11px;">{dt}</div>
            </div>"""
        cor = "#00C851" if (ret or 0) >= 0 else "#FF4444"
        seta= "▲" if (ret or 0) >= 0 else "▼"
        return f"""<div style="background:#313244;border-radius:12px;padding:14px 20px;
                   flex:1;min-width:180px;border-left:4px solid {cor};">
          <div style="color:#888;font-size:12px;">{label}</div>
          <div style="color:#CDD6F4;font-size:22px;font-weight:800;">R$ {val:,.2f}</div>
          <div style="color:{cor};font-size:14px;font-weight:700;">{seta} {ret:+.2f}%</div>
          <div style="color:#666;font-size:11px;">em {dt}</div>
        </div>"""

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;
         color:#CDD6F4;padding:24px;border-radius:16px;">

      <!-- Título -->
      <h1 style="color:#CBA6F7;margin:0 0 6px 0;">📦 Simulador em Lote — Resultado da Carteira</h1>
      <p style="color:#888;font-size:13px;margin-bottom:20px;">
        Compra em <b>{dt_entrada_str}</b> | R$ {valor_por:,.2f} por ativo |
        {len([r for r in resultados if r['status']=='✅ OK'])} ativos | Total investido:
        <b>R$ {total_inv:,.2f}</b>
      </p>

      <!-- Cards de Resumo do Portfólio -->
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
        <div style="background:#313244;border-radius:12px;padding:14px 20px;flex:1;min-width:180px;">
          <div style="color:#888;font-size:12px;">💼 Investimento Total</div>
          <div style="color:#CDD6F4;font-size:22px;font-weight:800;">R$ {total_inv:,.2f}</div>
          <div style="color:#666;font-size:11px;">em {dt_entrada_str}</div>
        </div>
        {card_resumo("📈 Portfólio T+1 Mês", total_val_1m, ret_port_1m, dt_1m_str)}
        {card_resumo("📈 Portfólio T+2 Meses", total_val_2m, ret_port_2m, dt_2m_str)}
        <div style="background:#313244;border-radius:12px;padding:14px 20px;flex:1;min-width:180px;">
          <div style="color:#888;font-size:12px;">⭐ Melhor Ativo (1M)</div>"""

    # Melhor e pior ativo em 1M
    ok_1m = [r for r in resultados if r.get("retorno_1m") is not None]
    if ok_1m:
        melhor = max(ok_1m, key=lambda x: x["retorno_1m"])
        pior   = min(ok_1m, key=lambda x: x["retorno_1m"])
        html += f"""
          <div style="color:#00C851;font-size:20px;font-weight:800;">{melhor['ticker']}</div>
          <div style="color:#00C851;font-size:13px;">{melhor['retorno_1m']:+.2f}%</div>
        </div>
        <div style="background:#313244;border-radius:12px;padding:14px 20px;flex:1;min-width:180px;">
          <div style="color:#888;font-size:12px;">💀 Pior Ativo (1M)</div>
          <div style="color:#FF4444;font-size:20px;font-weight:800;">{pior['ticker']}</div>
          <div style="color:#FF4444;font-size:13px;">{pior['retorno_1m']:+.2f}%</div>
        </div>"""
    else:
        html += """<div style="color:#888;">—</div></div>"""

    html += """
      </div>

      <hr style="border-color:#45475A;margin-bottom:20px;">

      <!-- Cards por ativo -->
      <h2 style="color:#89B4FA;margin:0 0 14px 0;font-size:16px;">
        📋 Resultado detalhado por ativo
      </h2>
      <div style="display:flex;flex-wrap:wrap;gap:12px;">"""

    for r in sorted(resultados, key=lambda x: (x.get("retorno_1m") or -9999), reverse=True):
        if r["status"] != "✅ OK":
            # Card de erro
            html += f"""
            <div style="background:#313244;border-radius:12px;padding:14px;
                 min-width:160px;border-left:4px solid #FF4444;">
              <div style="font-size:18px;font-weight:800;color:#FF4444;">{r['ticker']}</div>
              <div style="color:#888;font-size:12px;margin-top:4px;">{r['status']}</div>
            </div>"""
            continue

        ret1  = r.get("retorno_1m")
        ret2  = r.get("retorno_2m")
        val1  = r.get("valor_1m")
        val2  = r.get("valor_2m")
        cor1  = "#00C851" if (ret1 or 0) >= 0 else "#FF4444"
        cor2  = "#00C851" if (ret2 or 0) >= 0 else "#FF4444"
        s1    = "▲" if (ret1 or 0) >= 0 else "▼"
        s2    = "▲" if (ret2 or 0) >= 0 else "▼"
        borda = cor1 if ret1 is not None else "#45475A"

        html += f"""
            <div style="background:#313244;border-radius:12px;padding:16px;
                 min-width:170px;border-left:4px solid {borda};flex:1;">

              <div style="font-size:20px;font-weight:800;color:#CDD6F4;">{r['ticker']}</div>
              <div style="color:#888;font-size:11px;margin-bottom:10px;">
                Entrada: R$ {r['preco_entrada']:,.2f} em {r['data_entrada']}
                <br>{r['cotas']} cotas (frac.)
              </div>

              <!-- T+1M -->
              <div style="margin-bottom:8px;">
                <div style="color:#888;font-size:10px;">📅 T+1 Mês ({r['data_1m']})</div>"""
        if ret1 is not None:
            html += f"""
                <div style="color:#CDD6F4;font-size:13px;">R$ {r['preco_1m']:,.2f} / ação</div>
                <div style="color:{cor1};font-size:15px;font-weight:700;">{s1} {ret1:+.2f}%</div>
                <div style="color:#CDD6F4;font-size:12px;">Portfólio: R$ {val1:,.2f}</div>"""
        else:
            html += '<div style="color:#888;font-size:12px;">Aguardando...</div>'

        html += """
              </div>
              <hr style="border-color:#45475A;margin:8px 0;">
              <!-- T+2M -->
              <div>"""
        html += f'<div style="color:#888;font-size:10px;">📅 T+2 Meses ({r["data_2m"]})</div>'
        if ret2 is not None:
            html += f"""
                <div style="color:#CDD6F4;font-size:13px;">R$ {r['preco_2m']:,.2f} / ação</div>
                <div style="color:{cor2};font-size:15px;font-weight:700;">{s2} {ret2:+.2f}%</div>
                <div style="color:#CDD6F4;font-size:12px;">Portfólio: R$ {val2:,.2f}</div>"""
        else:
            html += '<div style="color:#888;font-size:12px;">Aguardando...</div>'

        html += """
              </div>
            </div>"""

    html += """
      </div>

      <hr style="border-color:#45475A;margin-top:20px;">
      <p style="color:#666;font-size:11px;">
        ⚠️ Simulação baseada em preços de fechamento ajustado. Não considera custos de corretagem,
        impostos ou slippage. Resultados passados não garantem resultados futuros.
      </p>
    </div>"""
    return html


displayHTML(gerar_painel_html(
    resultados,
    VALOR_POR_ACAO,
    DT_ENTRADA.strftime("%d/%m/%Y"),
    DT_1M.strftime("%d/%m/%Y"),
    DT_2M.strftime("%d/%m/%Y"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📈 Gráfico 1 — Evolução do Valor de Cada Ativo
# MAGIC
# MAGIC Este gráfico mostra a **curva de preço normalizada** de cada ativo ao longo do período.
# MAGIC Todos os ativos partem de **100** na data de entrada — facilitando a comparação visual.
# MAGIC
# MAGIC > 📌 Um valor de **110** significa que o ativo valorizou **+10%**.
# MAGIC > Um valor de **90** significa que o ativo desvalorizou **-10%**.

# COMMAND ----------

# DBTITLE 1,Gráfico 1 — Retorno normalizado (base 100)
def plot_retorno_normalizado(precos, resultados, dt_entrada, dt_1m, dt_2m, valor_por):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    # Paleta de cores cíclica
    CORES = [
        "#CBA6F7","#89B4FA","#A6E3A1","#FAB387","#F38BA8",
        "#94E2D5","#F9E2AF","#B4BEFE","#EBA0AC","#74C7EC",
    ]

    ativos_plotados = 0
    for i, r in enumerate(resultados):
        tk_raw = r.get("ticker_raw", r["ticker"] + ".SA")
        if tk_raw not in precos:
            continue
        df = precos[tk_raw]["Close"].sort_index()
        # Encontra o preço na data de entrada para normalizar
        ts_ent = pd.Timestamp(dt_entrada)
        idx_ent = df.index.searchsorted(ts_ent)
        if idx_ent >= len(df):
            continue
        preco_base = float(df.iloc[idx_ent])
        if preco_base <= 0:
            continue

        normalizado = df / preco_base * 100
        cor = CORES[i % len(CORES)]
        ax.plot(normalizado.index, normalizado.values,
                color=cor, linewidth=1.8, label=r["ticker"], alpha=0.9)

        # Ponto no T+1M
        ts_1m = pd.Timestamp(dt_1m)
        idx_1m = min(df.index.searchsorted(ts_1m), len(df) - 1)
        ax.scatter([df.index[idx_1m]], [float(normalizado.iloc[idx_1m])],
                   color=cor, s=60, zorder=5)

        ativos_plotados += 1

    # Linhas de referência temporal
    ax.axvline(pd.Timestamp(dt_entrada), color="#CBA6F7", linewidth=2,
               linestyle="--", label=f"Entrada ({dt_entrada.strftime('%d/%m/%Y')})", zorder=6)
    ax.axvline(pd.Timestamp(dt_1m), color="#FAB387", linewidth=1.5,
               linestyle=":", label=f"T+1M ({dt_1m.strftime('%d/%m/%Y')})", zorder=6)
    if dt_2m <= datetime.now():
        ax.axvline(pd.Timestamp(dt_2m), color="#A6E3A1", linewidth=1.5,
                   linestyle=":", label=f"T+2M ({dt_2m.strftime('%d/%m/%Y')})", zorder=6)

    # Linha de breakeven (100 = sem ganho nem perda)
    ax.axhline(100, color="#888", linewidth=0.8, linestyle="-", alpha=0.5, label="Break-even (100)")

    # Zona de lucro / prejuízo
    ax.fill_between([pd.Timestamp(dt_entrada), pd.Timestamp(precos[list(precos.keys())[0]].index[-1])],
                    100, 200, alpha=0.04, color="#00C851")
    ax.fill_between([pd.Timestamp(dt_entrada), pd.Timestamp(precos[list(precos.keys())[0]].index[-1])],
                    0, 100, alpha=0.04, color="#FF4444")

    # Formatação
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=35, color="#CDD6F4", fontsize=8)
    plt.yticks(color="#CDD6F4", fontsize=9)
    ax.set_ylabel("Valor Normalizado (100 = entrada)", color="#CDD6F4", fontsize=10)
    ax.set_xlabel("Data", color="#CDD6F4", fontsize=10)
    ax.set_title(
        f"Evolução Normalizada dos Ativos  |  Entrada: {dt_entrada.strftime('%d/%m/%Y')}  |  R$ {valor_por:,.2f}/ativo",
        color="#CDD6F4", fontsize=12, fontweight="bold", pad=10
    )
    ax.grid(color="#313244", linewidth=0.5, alpha=0.6)
    ax.spines[:].set_color("#45475A")
    legend = ax.legend(loc="upper left", framealpha=0.3, facecolor="#313244",
                       edgecolor="#45475A", labelcolor="#CDD6F4", fontsize=8,
                       ncol=2 if ativos_plotados > 6 else 1)

    plt.tight_layout(pad=1.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1E1E2E")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


img1 = plot_retorno_normalizado(PRECOS, resultados, DT_ENTRADA, DT_1M, DT_2M, VALOR_POR_ACAO)
displayHTML(f"""
<div style="background:#1E1E2E;padding:20px;border-radius:16px;margin-top:10px;">
  <h3 style="color:#CBA6F7;font-family:'Segoe UI',sans-serif;margin:0 0 12px 0;">
    📈 Retorno Normalizado (base 100 = data de entrada)
  </h3>
  <img src="data:image/png;base64,{img1}" style="width:100%;border-radius:8px;"/>
  <p style="color:#666;font-size:11px;font-family:sans-serif;margin-top:8px;">
    Cada linha representa um ativo. Base 100 = preço na data de entrada.
    Acima de 100 = lucro; abaixo de 100 = prejuízo.
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📊 Gráfico 2 — Comparação de Retornos (Barras)
# MAGIC
# MAGIC Gráfico de barras comparando o retorno percentual de cada ativo
# MAGIC nos dois marcos temporais (**T+1M** e **T+2M**) lado a lado.

# COMMAND ----------

# DBTITLE 1,Gráfico 2 — Barras comparativas de retorno
def plot_barras_retorno(resultados):
    ok = [r for r in resultados if r.get("retorno_1m") is not None or r.get("retorno_2m") is not None]
    if not ok:
        print("⚠️ Nenhum dado suficiente para o gráfico de barras.")
        return None

    tickers  = [r["ticker"] for r in ok]
    ret_1m   = [r.get("retorno_1m") or 0 for r in ok]
    ret_2m   = [r.get("retorno_2m") or 0 for r in ok]

    x      = np.arange(len(tickers))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(tickers) * 1.4), 6))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    # Barras T+1M
    bars1 = ax.bar(x - width/2, ret_1m, width, label="T+1 Mês",
                   color=["#00C851" if v >= 0 else "#FF4444" for v in ret_1m],
                   alpha=0.85, edgecolor="#1E1E2E", linewidth=0.5)

    # Barras T+2M
    bars2 = ax.bar(x + width/2, ret_2m, width, label="T+2 Meses",
                   color=["#89B4FA" if v >= 0 else "#FAB387" for v in ret_2m],
                   alpha=0.75, edgecolor="#1E1E2E", linewidth=0.5)

    # Rótulos de valor nas barras
    for bar, val in zip(bars1, ret_1m):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h >= 0 else -1.2),
                f"{val:+.1f}%", ha="center", va="bottom", color="#CDD6F4",
                fontsize=8, fontweight="bold")

    for bar, val in zip(bars2, ret_2m):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h >= 0 else -1.2),
                f"{val:+.1f}%", ha="center", va="bottom", color="#BAC2DE",
                fontsize=8)

    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, color="#CDD6F4", fontsize=10, fontweight="bold")
    ax.set_ylabel("Retorno (%)", color="#CDD6F4", fontsize=10)
    ax.set_title("Retorno por Ativo — T+1M e T+2M", color="#CDD6F4",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#CDD6F4")
    ax.spines[:].set_color("#45475A")
    ax.grid(axis="y", color="#313244", linewidth=0.5, alpha=0.6)

    # Legenda manual
    import matplotlib.patches as mpatches
    leg = [
        mpatches.Patch(color="#00C851", alpha=0.85, label="T+1M (ganho)"),
        mpatches.Patch(color="#FF4444", alpha=0.85, label="T+1M (perda)"),
        mpatches.Patch(color="#89B4FA", alpha=0.75, label="T+2M (ganho)"),
        mpatches.Patch(color="#FAB387", alpha=0.75, label="T+2M (perda)"),
    ]
    ax.legend(handles=leg, loc="upper right", framealpha=0.3,
              facecolor="#313244", edgecolor="#45475A", labelcolor="#CDD6F4", fontsize=9)

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1E1E2E")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


img2 = plot_barras_retorno(resultados)
if img2:
    displayHTML(f"""
    <div style="background:#1E1E2E;padding:20px;border-radius:16px;margin-top:10px;">
      <h3 style="color:#CBA6F7;font-family:'Segoe UI',sans-serif;margin:0 0 12px 0;">
        📊 Comparação de Retorno por Ativo (T+1M × T+2M)
      </h3>
      <img src="data:image/png;base64,{img2}" style="width:100%;border-radius:8px;"/>
      <p style="color:#666;font-size:11px;font-family:sans-serif;margin-top:8px;">
        Barras azuis/laranja = T+2M. Verde/vermelho = T+1M.
        Valores acima do eixo = lucro; abaixo = prejuízo.
      </p>
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📊 Gráfico 3 — Evolução do Portfólio Total
# MAGIC
# MAGIC Este gráfico mostra a **evolução do valor total** do portfólio ao longo do período —
# MAGIC como se você tivesse comprado todos os ativos ao mesmo tempo e acompanhado o valor diariamente.

# COMMAND ----------

# DBTITLE 1,Gráfico 3 — Valor total do portfólio ao longo do tempo
def plot_portfolio_total(precos, resultados, dt_entrada, dt_1m, dt_2m, valor_por):
    """
    Calcula o valor total do portfólio dia a dia e plota a evolução.
    O portfólio é a soma do valor de mercado de cada posição.
    """
    # Monta matriz de preços alinhados
    series_list = []
    for r in resultados:
        if r["status"] != "✅ OK":
            continue
        tk_raw = r.get("ticker_raw", r["ticker"] + ".SA")
        if tk_raw not in precos or r["preco_entrada"] is None:
            continue
        cotas = valor_por / r["preco_entrada"]
        valor_serie = precos[tk_raw]["Close"] * cotas
        series_list.append(valor_serie)

    if not series_list:
        print("⚠️ Sem dados suficientes para o gráfico de portfólio.")
        return None

    # Alinha todas as séries e soma dia a dia
    df_port = pd.concat(series_list, axis=1).dropna(how="all")
    portfolio_diario = df_port.sum(axis=1, min_count=1).dropna()

    # Total investido (só ativos com dados)
    total_investido = valor_por * len(series_list)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1E1E2E")
    fig.patch.set_facecolor("#1E1E2E")

    # ── Painel superior: valor absoluto do portfólio ──────────────────────────
    cor_area = "#00C851" if portfolio_diario.iloc[-1] >= total_investido else "#FF4444"
    ax1.plot(portfolio_diario.index, portfolio_diario.values,
             color="#CBA6F7", linewidth=2.5, zorder=4, label="Portfólio")
    ax1.fill_between(portfolio_diario.index, portfolio_diario.values,
                     total_investido, alpha=0.2,
                     where=(portfolio_diario >= total_investido), color="#00C851")
    ax1.fill_between(portfolio_diario.index, portfolio_diario.values,
                     total_investido, alpha=0.2,
                     where=(portfolio_diario < total_investido), color="#FF4444")
    ax1.axhline(total_investido, color="#FFBB33", linewidth=1.5,
                linestyle="--", label=f"Investido: R$ {total_investido:,.2f}", zorder=5)

    # Marcos temporais
    ax1.axvline(pd.Timestamp(dt_entrada), color="#CBA6F7", linewidth=2,
                linestyle="--", label=f"Entrada {dt_entrada.strftime('%d/%m/%Y')}", zorder=6)
    ax1.axvline(pd.Timestamp(dt_1m), color="#FAB387", linewidth=1.5,
                linestyle=":", label=f"T+1M {dt_1m.strftime('%d/%m/%Y')}", zorder=6)
    if dt_2m <= datetime.now():
        ax1.axvline(pd.Timestamp(dt_2m), color="#A6E3A1", linewidth=1.5,
                    linestyle=":", label=f"T+2M {dt_2m.strftime('%d/%m/%Y')}", zorder=6)

    # Pontos nos marcos
    for dt_marco, cor_marco, lbl in [
        (dt_1m, "#FAB387", "T+1M"),
        (dt_2m, "#A6E3A1", "T+2M"),
    ]:
        if dt_marco <= datetime.now():
            idx = portfolio_diario.index.searchsorted(pd.Timestamp(dt_marco))
            if idx < len(portfolio_diario):
                v = float(portfolio_diario.iloc[idx])
                ax1.scatter([portfolio_diario.index[idx]], [v], color=cor_marco,
                            s=100, zorder=8)
                ax1.annotate(f"R$ {v:,.0f}", (portfolio_diario.index[idx], v),
                             textcoords="offset points", xytext=(8, 8),
                             color=cor_marco, fontsize=9, fontweight="bold")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax1.set_ylabel("Valor do Portfólio (R$)", color="#CDD6F4", fontsize=10)
    ax1.set_title(
        f"Evolução do Portfólio Total  |  {len(series_list)} ativos  |  Investido: R$ {total_investido:,.2f}",
        color="#CDD6F4", fontsize=12, fontweight="bold", pad=8
    )
    ax1.tick_params(colors="#CDD6F4")
    ax1.grid(color="#313244", linewidth=0.5, alpha=0.6)
    ax1.spines[:].set_color("#45475A")
    ax1.legend(loc="upper left", framealpha=0.3, facecolor="#313244",
               edgecolor="#45475A", labelcolor="#CDD6F4", fontsize=8)

    # ── Painel inferior: retorno % acumulado ──────────────────────────────────
    ret_acum = (portfolio_diario / total_investido - 1) * 100
    ax2.plot(ret_acum.index, ret_acum.values, color="#89B4FA", linewidth=1.5)
    ax2.fill_between(ret_acum.index, ret_acum.values, 0,
                     where=(ret_acum >= 0), color="#00C851", alpha=0.2)
    ax2.fill_between(ret_acum.index, ret_acum.values, 0,
                     where=(ret_acum < 0), color="#FF4444", alpha=0.2)
    ax2.axhline(0, color="#888", linewidth=0.8)
    ax2.axvline(pd.Timestamp(dt_entrada), color="#CBA6F7", linewidth=1.5, linestyle="--", zorder=6)
    ax2.axvline(pd.Timestamp(dt_1m), color="#FAB387", linewidth=1, linestyle=":", zorder=6)
    if dt_2m <= datetime.now():
        ax2.axvline(pd.Timestamp(dt_2m), color="#A6E3A1", linewidth=1, linestyle=":", zorder=6)
    ax2.set_ylabel("Retorno %", color="#CDD6F4", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, color="#CDD6F4", fontsize=8)
    ax2.tick_params(colors="#CDD6F4")
    ax2.grid(color="#313244", linewidth=0.4, alpha=0.5)
    ax2.spines[:].set_color("#45475A")

    plt.tight_layout(pad=1.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1E1E2E")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


img3 = plot_portfolio_total(PRECOS, resultados, DT_ENTRADA, DT_1M, DT_2M, VALOR_POR_ACAO)
if img3:
    displayHTML(f"""
    <div style="background:#1E1E2E;padding:20px;border-radius:16px;margin-top:10px;">
      <h3 style="color:#CBA6F7;font-family:'Segoe UI',sans-serif;margin:0 0 12px 0;">
        💼 Evolução do Valor Total do Portfólio
      </h3>
      <img src="data:image/png;base64,{img3}" style="width:100%;border-radius:8px;"/>
      <p style="color:#666;font-size:11px;font-family:sans-serif;margin-top:8px;">
        Linha roxa = valor do portfólio | Linha amarela tracejada = valor investido (break-even)
        | Painel inferior = retorno % acumulado.
      </p>
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Resumo Final em Tabela

# COMMAND ----------

# DBTITLE 1,Tabela resumo final com formatação
def formatar_linha_resumo(r):
    """Retorna uma linha formatada para exibição."""
    return {
        "Ativo":           r["ticker"],
        "Preço Entrada":   f"R$ {r['preco_entrada']:,.2f}" if r.get("preco_entrada") else "—",
        "Cotas":           f"{r['cotas']:.4f}" if r.get("cotas") else "—",
        "Valor Inicial":   f"R$ {VALOR_POR_ACAO:,.2f}",
        "Preço T+1M":      f"R$ {r['preco_1m']:,.2f}" if r.get("preco_1m") else "Aguardando",
        "Valor T+1M":      f"R$ {r['valor_1m']:,.2f}" if r.get("valor_1m") else "—",
        "Retorno 1M":      f"{r['retorno_1m']:+.2f}%" if r.get("retorno_1m") is not None else "—",
        "Lucro/Prej. 1M":  f"R$ {r['valor_1m'] - VALOR_POR_ACAO:+,.2f}" if r.get("valor_1m") else "—",
        "Preço T+2M":      f"R$ {r['preco_2m']:,.2f}" if r.get("preco_2m") else "Aguardando",
        "Valor T+2M":      f"R$ {r['valor_2m']:,.2f}" if r.get("valor_2m") else "—",
        "Retorno 2M":      f"{r['retorno_2m']:+.2f}%" if r.get("retorno_2m") is not None else "—",
        "Lucro/Prej. 2M":  f"R$ {r['valor_2m'] - VALOR_POR_ACAO:+,.2f}" if r.get("valor_2m") else "—",
    }


linhas = [formatar_linha_resumo(r) for r in resultados if r["status"] == "✅ OK"]

if linhas:
    df_final = pd.DataFrame(linhas)

    # Linha de totais
    ok_r = [r for r in resultados if r["status"] == "✅ OK"]
    total_inv  = VALOR_POR_ACAO * len(ok_r)
    total_1m   = sum(r["valor_1m"] for r in ok_r if r.get("valor_1m") is not None)
    total_2m   = sum(r["valor_2m"] for r in ok_r if r.get("valor_2m") is not None)
    ret_tot_1m = (total_1m - total_inv) / total_inv * 100 if total_1m and total_inv > 0 else None
    ret_tot_2m = (total_2m - total_inv) / total_inv * 100 if total_2m and total_inv > 0 else None

    total_row = {
        "Ativo":          "📦 TOTAL",
        "Preço Entrada":  "—",
        "Cotas":          "—",
        "Valor Inicial":  f"R$ {total_inv:,.2f}",
        "Preço T+1M":     "—",
        "Valor T+1M":     f"R$ {total_1m:,.2f}" if total_1m else "—",
        "Retorno 1M":     f"{ret_tot_1m:+.2f}%" if ret_tot_1m is not None else "—",
        "Lucro/Prej. 1M": f"R$ {total_1m - total_inv:+,.2f}" if total_1m else "—",
        "Preço T+2M":     "—",
        "Valor T+2M":     f"R$ {total_2m:,.2f}" if total_2m else "—",
        "Retorno 2M":     f"{ret_tot_2m:+.2f}%" if ret_tot_2m is not None else "—",
        "Lucro/Prej. 2M": f"R$ {total_2m - total_inv:+,.2f}" if total_2m else "—",
    }

    df_final = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    display(df_final)

    # Salva CSV
    try:
        nome_csv = f"/tmp/simulador_lote_{DATA_ENTRADA.replace('-','')}.csv"
        df_final.to_csv(nome_csv, index=False)
        print(f"\n💾 Tabela salva em: {nome_csv}")
    except Exception as e:
        print(f"\n⚠️ Não foi possível salvar CSV: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🏁 Conclusão
# MAGIC
# MAGIC ### O que este notebook calculou:
# MAGIC
# MAGIC | Etapa | O que foi feito |
# MAGIC |-------|-----------------|
# MAGIC | ① | Leu a configuração: tickers, valor e data |
# MAGIC | ② | Baixou preços históricos via Yahoo Finance |
# MAGIC | ③ | Calculou quantas cotas seriam compradas na data de entrada |
# MAGIC | ④ | Buscou os preços nos marcos T+1M e T+2M |
# MAGIC | ⑤ | Calculou o retorno de cada ativo e do portfólio total |
# MAGIC | ⑥ | Exibiu painel visual com cards por ativo |
# MAGIC | ⑦ | Gerou 3 gráficos: normalizado, barras comparativas e portfólio total |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚙️ Como alterar a simulação?
# MAGIC
# MAGIC Basta editar a **primeira célula de configuração** e executar o notebook novamente:
# MAGIC
# MAGIC ```python
# MAGIC DATA_ENTRADA   = "2024-06-01"   # mude a data
# MAGIC VALOR_POR_ACAO = 5000.00        # mude o valor
# MAGIC TICKERS = ["WEGE3.SA", "PRIO3.SA", "RDOR3.SA"]  # mude os ativos
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > ⚠️ **Aviso Legal:** Esta simulação é puramente educacional e retrospectiva.
# MAGIC > Resultados passados não garantem resultados futuros.
# MAGIC > Não considera custos de corretagem, impostos (IOF, IR) ou liquidez real do mercado.
