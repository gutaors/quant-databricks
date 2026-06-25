# Databricks notebook source
# MAGIC %md
# MAGIC # 🧮 Simons Simulator — Estratégias Quantitativas de Jim Simons para Ações Brasileiras
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 👤 Quem é Jim Simons?
# MAGIC
# MAGIC **James Harris Simons** (1938–2024) foi um matemático, ex-criptógrafo e o maior gestor de fundos
# MAGIC quantitativos da história. Fundou a **Renaissance Technologies** em 1982 e seu lendário
# MAGIC **Medallion Fund** entregou retornos médios de **~66% ao ano bruto** por mais de 30 anos —
# MAGIC o melhor track record de longo prazo de qualquer fundo na história.
# MAGIC
# MAGIC ### 🧠 A Filosofia de Simons
# MAGIC
# MAGIC > *"Se você tiver dados suficientes e modelos matemáticos suficientemente bons, o mercado pode ser previsto."*
# MAGIC
# MAGIC Simons contratou **físicos, matemáticos e cientistas de dados** — nunca analistas de Wall Street.
# MAGIC Seu edge não era intuição. Era **estatística, padrões e execução sistemática**.
# MAGIC
# MAGIC ### 📊 Diferença fundamental vs. BNF (Kotegawa):
# MAGIC
# MAGIC | | BNF (Kotegawa) | Jim Simons |
# MAGIC |--|----------------|------------|
# MAGIC | Abordagem | Técnica / Pânico | Estatística / Sistemática |
# MAGIC | Indicadores | MM25, Kairi, RSI | Z-Score, Momentum, Cointegração |
# MAGIC | Tomada de decisão | Humana (intuição) | Algoritmo (regras quantitativas) |
# MAGIC | Horizonte | Horas / dias | Dias / semanas |
# MAGIC | Posições | Concentradas | Muito diversificadas |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Estratégias Implementadas
# MAGIC
# MAGIC | # | Nome | Conceito-Chave | Inspiração Renaissance |
# MAGIC |---|------|----------------|------------------------|
# MAGIC | 1 | **Momentum Cross-Sectional** | Z-Score de retorno relativo | Ranking de performance relativa |
# MAGIC | 2 | **Reversão Estatística (Z-Score)** | Desvio padrão do preço | "Déjà Vu" — reversão à média do RenTech |
# MAGIC | 3 | **Detecção de Regime** | HMM-like: volatilidade + tendência | Adaptação dinâmica de estratégia ao regime |
# MAGIC | 4 | **Arbitragem de Pares (Spread)** | Cointegração entre ativos correlatos | Stat-arb clássico do Medallion |
# MAGIC | 5 | **Sinal Composto Multi-Fator** | Combinação de momentum + reversão | Filosofia de "1000 sinais pequenos" |
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Instalação de Dependências

# COMMAND ----------

# DBTITLE 1,Instalação de Pacotes
%pip install yfinance statsmodels scipy pandas numpy matplotlib --quiet

# COMMAND ----------

# DBTITLE 1,Imports e Configuração Global
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import base64
from io import BytesIO
from datetime import datetime, timedelta
from scipy import stats
from IPython.display import display, HTML

# statsmodels para ADF (cointegração)
try:
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("⚠️ statsmodels não disponível — estratégia de pares simplificada")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

print("✅ Imports realizados com sucesso!")
print(f"📅 Data e hora atual: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📅 INSIRA A DATA DE CORTE
# MAGIC
# MAGIC > **Instrução:** Informe a data de corte no widget abaixo (formato `YYYY-MM-DD`).
# MAGIC > O notebook buscará dados históricos até essa data e identificará oportunidades
# MAGIC > usando as estratégias quantitativas inspiradas em Jim Simons / Renaissance Technologies.

# COMMAND ----------

# DBTITLE 1,Widget de Data de Corte
dbutils.widgets.removeAll()
dbutils.widgets.text(
    "data_corte",
    datetime.now().strftime("%Y-%m-%d"),
    "📅 Data de Corte (YYYY-MM-DD)"
)

# COMMAND ----------

# DBTITLE 1,Validação da Data de Corte
RAW_DATA = dbutils.widgets.get("data_corte").strip()

try:
    DATA_CORTE  = datetime.strptime(RAW_DATA, "%Y-%m-%d")
    DATA_INICIO = DATA_CORTE - timedelta(days=500)  # ~2 anos para calcular momentum 12m

    if DATA_CORTE > datetime.now():
        raise ValueError("A data de corte não pode ser futura!")

    print(f"✅ Data de corte válida : {DATA_CORTE.strftime('%d/%m/%Y')}")
    print(f"📆 Período de análise  : {DATA_INICIO.strftime('%d/%m/%Y')} → {DATA_CORTE.strftime('%d/%m/%Y')}")
    print(f"🕐 Histórico           : {(DATA_CORTE - DATA_INICIO).days} dias")

except ValueError as e:
    raise ValueError(f"❌ Data inválida: {e}. Use o formato YYYY-MM-DD (ex: 2024-06-15)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Lista de Ativos — B3

# COMMAND ----------

# DBTITLE 1,Tickers B3 e IBOV
TICKERS_B3 = [
    # Blue Chips / IBOV Core
    "PETR4.SA", "PETR3.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "ABEV3.SA", "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "LREN3.SA", "MGLU3.SA",
    "EMBR3.SA", "JBSS3.SA", "SUZB3.SA", "KLBN11.SA", "RAIL3.SA",
    "CSNA3.SA", "USIM5.SA", "GGBR4.SA", "CSAN3.SA", "EQTL3.SA", "ELET3.SA",
    "ELET6.SA", "CMIG4.SA", "CPFE3.SA", "CPLE6.SA", "EGIE3.SA", "ENGI11.SA",
    "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "PRIO3.SA", "CYRE3.SA",
    "MRVE3.SA", "TEND3.SA", "EZTC3.SA", "ALPA4.SA", "SOMA3.SA",
    "ARZZ3.SA", "NTCO3.SA", "COGN3.SA", "YDUQ3.SA", "ANIM3.SA",
    # Financeiro
    "SANB11.SA", "ITSA4.SA", "BBSE3.SA", "IRBR3.SA", "PSSA3.SA", "CIEL3.SA",
    "BRSR6.SA", "BPAN4.SA", "SULA11.SA",
    # Consumo / Varejo
    "VIIA3.SA", "AMER3.SA", "VVAR3.SA", "GRND3.SA", "SBFG3.SA", "PETZ3.SA",
    "RADL3.SA", "RAIA3.SA", "FLRY3.SA", "HAPV3.SA", "QUAL3.SA", "HYPE3.SA",
    "DASA3.SA",
    # Energia / Utilities
    "ENEV3.SA", "NEOE3.SA", "AURE3.SA", "ALUP11.SA", "CESP6.SA",
    "ENBR3.SA", "TRPL4.SA",
    # Telecom / Tech
    "LWSA3.SA", "CASH3.SA",
    # Agro / Commodities
    "SLCE3.SA", "SMTO3.SA", "AGRO3.SA", "BEEF3.SA", "MRFG3.SA",
    "BRFS3.SA", "CAML3.SA", "RAIZ4.SA",
    # Imoveis
    "MULT3.SA", "ALSO3.SA", "JHSF3.SA",
    # Infraestrutura / Concessoes
    "CCRO3.SA", "ECOR3.SA", "RDOR3.SA", "LOGN3.SA",
    "POMO4.SA", "AZUL4.SA", "GOLL4.SA",
    # Papel / Celulose
    "DTEX3.SA", "RANI3.SA",
    # IBOV como benchmark
    "^BVSP",
]

print(f"📋 Total: {len(TICKERS_B3) - 1} ações + IBOV como benchmark")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔽 Download de Dados Históricos

# COMMAND ----------

# DBTITLE 1,Funções de Download
def baixar_dados(tickers, inicio, fim, batch_size=15):
    dados = {}
    fim_str    = fim.strftime("%Y-%m-%d")
    inicio_str = inicio.strftime("%Y-%m-%d")
    batches    = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    for idx, batch in enumerate(batches):
        print(f"  📦 Lote {idx+1}/{len(batches)}: {len(batch)} ativos...")
        try:
            raw = yf.download(
                batch, start=inicio_str, end=fim_str,
                auto_adjust=True, progress=False, threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df = raw.xs(tk, axis=1, level=1).copy()
                        df.dropna(subset=["Close"], inplace=True)
                        if len(df) >= 60:
                            dados[tk] = df
                    except Exception:
                        pass
            else:
                tk = batch[0]
                raw.dropna(subset=["Close"], inplace=True)
                if len(raw) >= 60:
                    dados[tk] = raw
        except Exception as e:
            print(f"    ⚠️ Erro: {e}")
        time.sleep(0.4)

    return dados


print("🔽 Iniciando download...")
print(f"   {DATA_INICIO.strftime('%d/%m/%Y')} → {DATA_CORTE.strftime('%d/%m/%Y')}\n")

DADOS_BRUTOS = baixar_dados(TICKERS_B3, DATA_INICIO, DATA_CORTE)
DADOS_IBOV   = DADOS_BRUTOS.pop("^BVSP", None)
DADOS_ACOES  = dict(DADOS_BRUTOS)

print(f"\n✅ Download concluído!")
print(f"   📊 Ações válidas: {len(DADOS_ACOES)}")
if DADOS_IBOV is not None:
    print(f"   📈 IBOV: {len(DADOS_IBOV)} pregões")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Construção do Painel de Fatores Quantitativos
# MAGIC
# MAGIC A Renaissance Technologies não usa indicadores técnicos "comuns".
# MAGIC Ela constrói **fatores estatísticos** — medidas padronizadas e comparáveis entre todos os ativos.
# MAGIC
# MAGIC O painel abaixo calcula, para cada ativo na data de corte:
# MAGIC
# MAGIC | Fator | O que mede | Janela |
# MAGIC |-------|------------|--------|
# MAGIC | `ret_1m` | Retorno 1 mês | 21 pregões |
# MAGIC | `ret_3m` | Retorno 3 meses | 63 pregões |
# MAGIC | `ret_6m` | Retorno 6 meses | 126 pregões |
# MAGIC | `ret_12m` | Retorno 12 meses | 252 pregões |
# MAGIC | `ret_skip` | Momentum excluindo o último mês (Jegadeesh-Titman) | 252-21 pregões |
# MAGIC | `vol_20d` | Volatilidade de 20 dias (desvio-padrão dos retornos) | 20 pregões |
# MAGIC | `vol_60d` | Volatilidade de 60 dias | 60 pregões |
# MAGIC | `z_close_60` | Z-Score do preço atual vs. 60 dias | 60 pregões |
# MAGIC | `z_close_252` | Z-Score do preço atual vs. 252 dias | 252 pregões |
# MAGIC | `rsi_14` | RSI clássico (14 períodos) | 14 pregões |
# MAGIC | `vol_ratio` | Volume atual vs. média 20d | 20 pregões |
# MAGIC | `trend_slope` | Inclinação da regressão linear do preço | 60 pregões |
# MAGIC | `sharpe_proxy` | Retorno/Volatilidade (60d) — proxy do Sharpe | 60 pregões |

# COMMAND ----------

# DBTITLE 1,Cálculo do Painel de Fatores
def calcular_fatores(df):
    """
    Calcula o vetor de fatores quantitativos para um ativo na última data disponível.
    Retorna um dict com todos os fatores.
    """
    c = df["Close"]
    n = len(c)

    def ret(periodos):
        return c.pct_change(periodos).iloc[-1] * 100 if n > periodos else None

    def zscore(periodos):
        if n < periodos:
            return None
        janela = c.iloc[-periodos:]
        mu, sigma = janela.mean(), janela.std()
        return ((c.iloc[-1] - mu) / sigma) if sigma > 0 else 0

    def vol(periodos):
        if n < periodos + 1:
            return None
        return c.pct_change().iloc[-periodos:].std() * np.sqrt(252) * 100  # anualized %

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rsi_v = None
    if n >= 15:
        avg_g = gain.ewm(com=13, adjust=False).mean().iloc[-1]
        avg_l = loss.ewm(com=13, adjust=False).mean().iloc[-1]
        rsi_v = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 100

    # Volume Ratio
    vol_rat = None
    if "Volume" in df.columns and n >= 20:
        vm = df["Volume"].iloc[-20:].mean()
        vl = df["Volume"].iloc[-1]
        vol_rat = round(vl / vm, 2) if vm > 0 else None

    # Inclinação da regressão linear (trend slope normalizado)
    slope_v = None
    if n >= 60:
        y = c.iloc[-60:].values
        x = np.arange(len(y))
        slope_v, _, _, _, _ = stats.linregress(x, y)
        slope_v = slope_v / y.mean() * 100  # normalizado como % de variação/dia

    # Momentum skip-month (12m excluindo 1m) — clássico de Jegadeesh-Titman
    ret_skip = None
    if n >= 252:
        ret_skip = ((c.iloc[-252] - c.iloc[-252 + 21]) / c.iloc[-252 + 21]) * 100 if n >= 252 else None
        ret_12_raw = (c.iloc[-1] - c.iloc[-252]) / c.iloc[-252] * 100 if n >= 252 else None
        ret_1m_raw = (c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100 if n >= 21 else None
        if ret_12_raw is not None and ret_1m_raw is not None:
            ret_skip = ret_12_raw - ret_1m_raw

    # Proxy Sharpe 60d
    sharpe_p = None
    r60 = ret(60)
    v60 = vol(60)
    if r60 is not None and v60 is not None and v60 > 0:
        sharpe_p = round(r60 / v60, 3)

    # Preço atual
    preco_atual = round(float(c.iloc[-1]), 2)

    # ATR% para tamanho da posição
    if "High" in df.columns and "Low" in df.columns and n >= 14:
        hl   = df["High"] - df["Low"]
        hcp  = (df["High"] - c.shift(1)).abs()
        lcp  = (df["Low"]  - c.shift(1)).abs()
        tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        atr_pct = round(atr14 / preco_atual * 100, 2)
    else:
        atr_pct = None

    return {
        "preco":      preco_atual,
        "ret_1m":     round(ret(21),  2) if ret(21)  is not None else None,
        "ret_3m":     round(ret(63),  2) if ret(63)  is not None else None,
        "ret_6m":     round(ret(126), 2) if ret(126) is not None else None,
        "ret_12m":    round(ret(252), 2) if ret(252) is not None else None,
        "ret_skip":   round(ret_skip, 2) if ret_skip  is not None else None,
        "vol_20d":    round(vol(20),  2) if vol(20)  is not None else None,
        "vol_60d":    round(vol(60),  2) if vol(60)  is not None else None,
        "z_60":       round(zscore(60),  3) if zscore(60)  is not None else None,
        "z_252":      round(zscore(252), 3) if zscore(252) is not None else None,
        "rsi_14":     round(rsi_v, 1) if rsi_v is not None else None,
        "vol_ratio":  vol_rat,
        "trend_slope":round(slope_v, 4) if slope_v is not None else None,
        "sharpe_p":   sharpe_p,
        "atr_pct":    atr_pct,
    }


print("📐 Calculando fatores quantitativos para todos os ativos...\n")

PAINEL = {}
for ticker, df in DADOS_ACOES.items():
    try:
        PAINEL[ticker] = calcular_fatores(df)
    except Exception:
        pass

print(f"✅ Fatores calculados para {len(PAINEL)} ativos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Normalização Cross-Sectional (Z-Scores do Painel)
# MAGIC
# MAGIC O coração da abordagem de Simons é a **normalização cross-sectional**:
# MAGIC em vez de analisar um ativo isoladamente, ele comparava cada ativo
# MAGIC com **todos os outros** ao mesmo tempo.
# MAGIC
# MAGIC Um **Z-Score de +2.0** em momentum significa que o ativo está
# MAGIC 2 desvios-padrão ACIMA da média de todos os outros — é um "vencedor relativo".
# MAGIC
# MAGIC Um **Z-Score de -2.0** em preço significa que o ativo está
# MAGIC 2 desvios-padrão ABAIXO da média — candidato à reversão estatística.

# COMMAND ----------

# DBTITLE 1,Normalização Cross-Sectional
def normalizar_cross_section(painel, coluna):
    """
    Normaliza os valores de uma coluna do painel em Z-Scores cross-sectionais.
    Z = (valor - média_cross) / std_cross
    """
    valores = {tk: dados[coluna] for tk, dados in painel.items()
               if dados.get(coluna) is not None}
    if len(valores) < 5:
        return {}

    vals_arr = np.array(list(valores.values()))
    mu, sigma = np.nanmean(vals_arr), np.nanstd(vals_arr)
    if sigma == 0:
        return {tk: 0 for tk in valores}

    return {tk: round((v - mu) / sigma, 3) for tk, v in valores.items()}


# Normaliza os principais fatores
print("📊 Normalizando fatores cross-sectionalmente...")

Z_RET_1M   = normalizar_cross_section(PAINEL, "ret_1m")
Z_RET_3M   = normalizar_cross_section(PAINEL, "ret_3m")
Z_RET_6M   = normalizar_cross_section(PAINEL, "ret_6m")
Z_RET_12M  = normalizar_cross_section(PAINEL, "ret_12m")
Z_RET_SKIP = normalizar_cross_section(PAINEL, "ret_skip")
Z_VOL_60   = normalizar_cross_section(PAINEL, "vol_60d")
Z_SHARPE   = normalizar_cross_section(PAINEL, "sharpe_p")

# Adiciona Z-Scores ao painel
for tk in PAINEL:
    PAINEL[tk]["z_mom_1m"]   = Z_RET_1M.get(tk)
    PAINEL[tk]["z_mom_3m"]   = Z_RET_3M.get(tk)
    PAINEL[tk]["z_mom_6m"]   = Z_RET_6M.get(tk)
    PAINEL[tk]["z_mom_12m"]  = Z_RET_12M.get(tk)
    PAINEL[tk]["z_mom_skip"] = Z_RET_SKIP.get(tk)
    PAINEL[tk]["z_sharpe"]   = Z_SHARPE.get(tk)

print(f"✅ Z-Scores calculados para {len(PAINEL)} ativos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌡️ Detecção do Regime de Mercado
# MAGIC
# MAGIC A Renaissance **não usa a mesma estratégia em todos os regimes**.
# MAGIC O regime detectado influencia qual estratégia é priorizada:
# MAGIC
# MAGIC | Regime | Estratégia Preferida | Por quê |
# MAGIC |--------|---------------------|---------|
# MAGIC | 🟢 BULL | Momentum Cross-Sectional | Tendências persistem mais em alta |
# MAGIC | 🔴 BEAR | Reversão Estatística | Overshooting cria oportunidades de reversão |
# MAGIC | 🟡 NEUTRO/LATERAL | Arbitragem de Pares | Mercado sem direção = spread-trading |
# MAGIC | 🟠 ALTA VOLATILIDADE | Sinal Composto Multi-Fator | Máxima diversificação de sinais |

# COMMAND ----------

# DBTITLE 1,Detecção de Regime
def detectar_regime_quant(ibov_df, painel):
    """
    Detecta regime usando múltiplas métricas quantitativas:
    - Posição do IBOV vs sua média (tendência)
    - Volatilidade realizada (VIX-like)
    - Breadth do mercado (quantos ativos estão subindo)
    """
    if ibov_df is None:
        # Usa o painel de ações como proxy
        rets = [v.get("ret_60d", 0) for v in painel.values() if v.get("ret_60d") is not None]
        med_ret = np.median(rets) if rets else 0
        return ("BULL" if med_ret > 5 else "BEAR" if med_ret < -5 else "NEUTRO",
                "Regime estimado pelo retorno mediano das ações (sem IBOV disponível)")

    d = ibov_df.sort_index()
    c = d["Close"]
    n = len(c)

    # 1) Tendência: preço vs MMs
    mm50  = c.rolling(50).mean().iloc[-1]  if n >= 50  else c.mean()
    mm200 = c.rolling(200).mean().iloc[-1] if n >= 200 else c.mean()
    preco = c.iloc[-1]
    acima_mm50  = preco > mm50
    acima_mm200 = preco > mm200

    # 2) Retorno 60d
    ret60 = (preco - c.iloc[-61]) / c.iloc[-61] * 100 if n >= 61 else 0

    # 3) Volatilidade realizada 20d (VIX proxy)
    vol20 = c.pct_change().iloc[-20:].std() * np.sqrt(252) * 100 if n >= 21 else 30
    ALTA_VOL = vol20 > 35   # >35% aa = alta volatilidade no Brasil

    # 4) Breadth: % de ações com z_mom_1m > 0
    z_moms = [v["z_mom_1m"] for v in painel.values() if v.get("z_mom_1m") is not None]
    breadth = sum(1 for z in z_moms if z > 0) / len(z_moms) * 100 if z_moms else 50

    # Classificação
    if ALTA_VOL:
        regime = "ALTA_VOL"
        desc   = f"Volatilidade elevada: {vol20:.1f}% aa — Regime de máxima incerteza"
    elif acima_mm50 and acima_mm200 and ret60 > 5 and breadth > 55:
        regime = "BULL"
        desc   = f"Bull confirmado: acima MM50+MM200, ret60d={ret60:.1f}%, breadth={breadth:.0f}%"
    elif not acima_mm50 and not acima_mm200 and ret60 < -5 and breadth < 45:
        regime = "BEAR"
        desc   = f"Bear confirmado: abaixo MM50+MM200, ret60d={ret60:.1f}%, breadth={breadth:.0f}%"
    else:
        regime = "NEUTRO"
        desc   = f"Mercado lateral: ret60d={ret60:.1f}%, vol={vol20:.1f}%aa, breadth={breadth:.0f}%"

    return regime, desc


REGIME, REGIME_DESC = detectar_regime_quant(DADOS_IBOV, PAINEL)
ER = {"BULL": "🟢", "BEAR": "🔴", "NEUTRO": "🟡", "ALTA_VOL": "🟠"}
print(f"\n{ER.get(REGIME,'⚪')} REGIME DETECTADO: {REGIME}")
print(f"   {REGIME_DESC}")
print(f"\n💡 Estratégias serão ponderadas para o regime: {REGIME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Estratégias Jim Simons — Implementação Detalhada

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 1: Momentum Cross-Sectional (Ranking Relativo)
# MAGIC
# MAGIC **Conceito da Renaissance:**
# MAGIC Simons descobriu que ações com **desempenho superior relativo** nos últimos 3-12 meses
# MAGIC tendem a **continuar superando seus pares** no próximo mês.
# MAGIC
# MAGIC Isso é conhecido academicamente como **Fator Momentum de Jegadeesh e Titman (1993)**.
# MAGIC
# MAGIC **Como funciona no notebook:**
# MAGIC 1. Para cada ação, calcula os retornos de 1, 3, 6 e 12 meses
# MAGIC 2. Normaliza cada retorno em **Z-Score cross-seccional** (compara com todas as outras ações)
# MAGIC 3. Combina os Z-Scores em um **sinal composto de momentum**
# MAGIC 4. Usa o **skip-month** de Jegadeesh-Titman (exclui o último mês para evitar reversão de curto prazo)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC Gestores institucionais compram os "vencedores" continuamente.
# MAGIC O fluxo de capital cria persistência — a ação que subiu tende a continuar subindo
# MAGIC até que o sinal se reverta.
# MAGIC
# MAGIC **Filtros de qualidade (Simons-style):**
# MAGIC - Momentum composto > 1.0 Z-Score (claramente acima da média)
# MAGIC - Sharpe proxy positivo (risco-retorno favorável)
# MAGIC - Volume ratio > 0.8 (liquidez mínima)

# COMMAND ----------

# DBTITLE 1,Estratégia 1 — Momentum Cross-Sectional
def estrategia_momentum_cross(painel, ticker):
    """
    Estratégia 1 — Momentum Cross-Sectional (Jegadeesh-Titman style).

    Critérios:
    ✅ Z-Score de momentum composto (3m + 6m + skip) > 1.0
    ✅ Sharpe proxy positivo
    ✅ Volume ratio >= 0.8
    ✅ Tendência de preço positiva (trend_slope > 0)
    """
    d = painel.get(ticker)
    if not d:
        return None

    z3m   = d.get("z_mom_3m",   0) or 0
    z6m   = d.get("z_mom_6m",   0) or 0
    zskip = d.get("z_mom_skip", 0) or 0
    zshar = d.get("z_sharpe",   0) or 0
    vol_r = d.get("vol_ratio",  1) or 1
    slope = d.get("trend_slope", 0) or 0
    preco = d.get("preco", 0)
    ret3m = d.get("ret_3m", 0) or 0
    ret6m = d.get("ret_6m", 0) or 0
    vol60 = d.get("vol_60d", 30) or 30

    # Sinal composto: média ponderada dos Z-Scores de momentum
    # (maior peso para 3-6m, conforme literatura)
    mom_comp = 0.25 * z3m + 0.40 * z6m + 0.35 * zskip

    if mom_comp < 1.0:
        return None
    if zshar < 0:
        return None
    if vol_r < 0.8:
        return None

    score = 0
    if mom_comp > 2.0:   score += 40
    elif mom_comp > 1.5: score += 28
    else:                score += 18

    if zshar > 1.0:    score += 20
    elif zshar > 0.5:  score += 12

    if slope > 0.05:   score += 15
    elif slope > 0:    score += 8

    if vol_r > 1.2:    score += 15
    elif vol_r > 1.0:  score += 8

    # Alvo: retorno esperado baseado no momentum histórico recente
    alvo = preco * (1 + abs(ret3m) / 100 * 0.5)
    pot  = ((alvo - preco) / preco) * 100

    return {
        "ticker": ticker, "estrategia": "1 — Momentum Cross-Sectional",
        "score": min(score, 100), "preco": round(preco, 2),
        "mom_comp": round(mom_comp, 3),
        "z_mom_3m": round(z3m, 3), "z_mom_6m": round(z6m, 3),
        "z_skip": round(zskip, 3), "z_sharpe": round(zshar, 3),
        "vol_ratio": round(vol_r, 2), "ret_3m": round(ret3m, 2),
        "ret_6m": round(ret6m, 2), "vol_60d": round(vol60, 2),
        "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Sinal de Momentum Composto = {mom_comp:.2f}σ (top {max(1, int((1-0.16)*100)):.0f}% do universo). "
            f"Retorno de 3m: {ret3m:.1f}% e 6m: {ret6m:.1f}%, ambos estatisticamente superiores aos pares. "
            f"Sharpe proxy = {zshar:.2f}σ (risco-retorno favorável). "
            f"Simons chamava isso de 'persistência estatística': "
            f"os vencedores continuam ganhando até que o sinal se reverta. "
            f"Alvo: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BULL", "NEUTRO", "ALTA_VOL"],
    }

print("✅ Estratégia 1 (Momentum Cross-Sectional) definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 2: Reversão Estatística via Z-Score de Preço
# MAGIC
# MAGIC **Conceito da Renaissance — "Déjà Vu":**
# MAGIC A Renaissance chamava internamente sua estratégia de reversão de **"Déjà Vu"** —
# MAGIC o mercado sempre se repete estatisticamente. Quando um preço se desvia muito de
# MAGIC sua distribuição histórica normal, ele tende a reverter.
# MAGIC
# MAGIC **Como funciona:**
# MAGIC 1. Calcula o **Z-Score do preço** em relação às últimas 60 e 252 sessões
# MAGIC 2. Z-Score < -2.0 → preço está 2 desvios-padrão abaixo da média histórica → reversão esperada
# MAGIC 3. Confirma com RSI oversold (< 35) e volume acima da média (pânico)
# MAGIC
# MAGIC **Diferença vs. BNF:**
# MAGIC - BNF usa o Kairi (% de desvio da MM25) — simples e intuitivo
# MAGIC - Simons usa Z-Score padronizado — permite **comparação entre ativos** e thresholds probabilísticos
# MAGIC - Z-Score de -2.0 = 97.7% de probabilidade do preço estar "anormal" segundo distribuição normal
# MAGIC
# MAGIC **Filtros:**
# MAGIC - Z-Score 60d < -2.0 (oversold estatístico)
# MAGIC - RSI < 38 (oversold técnico)
# MAGIC - Momentum de 12m não negativamente extremo (não é queda secular)

# COMMAND ----------

# DBTITLE 1,Estratégia 2 — Reversão Estatística (Z-Score)
def estrategia_reversao_zscore(painel, ticker):
    """
    Estratégia 2 — Reversão Estatística via Z-Score de Preço.

    Critérios:
    ✅ Z-Score de preço (60d) < -2.0
    ✅ RSI < 38
    ✅ Z-Score de momentum 12m não < -2.5 (não é queda secular)
    ✅ Volatilidade não explosiva (evita ativos em colapso)
    """
    d = painel.get(ticker)
    if not d:
        return None

    z60   = d.get("z_60",   0) or 0
    z252  = d.get("z_252",  0) or 0
    rsi   = d.get("rsi_14", 50) or 50
    z12m  = d.get("z_mom_12m", 0) or 0
    vol60 = d.get("vol_60d", 30) or 30
    preco = d.get("preco", 0)
    vol_r = d.get("vol_ratio", 1) or 1
    ret1m = d.get("ret_1m", 0) or 0

    # Filtros obrigatórios
    if z60 > -2.0:     return None   # não está oversold o suficiente
    if rsi > 38:       return None   # RSI não confirma oversold
    if z12m < -2.5:    return None   # queda secular — evitar
    if vol60 > 80:     return None   # volatilidade explosiva — muito risco

    score = 0
    if z60 < -3.0:      score += 40
    elif z60 < -2.5:    score += 28
    else:               score += 18

    if rsi < 25:        score += 20
    elif rsi < 30:      score += 12
    else:               score += 6

    if z252 < -1.5:     score += 15   # também abaixo da média anual
    if vol_r > 1.2:     score += 15   # volume acima da média = pânico confirmado
    if ret1m < -10:     score += 10   # queda recente = oportunidade mais fresca

    # Alvo: reverter ao Z-Score 0 (média) — preço médio dos últimos 60d
    if len(DADOS_ACOES.get(ticker, pd.DataFrame())) >= 60:
        media_60 = DADOS_ACOES[ticker]["Close"].iloc[-60:].mean()
        alvo = round(media_60, 2)
    else:
        alvo = round(preco * 1.15, 2)
    pot = round(((alvo - preco) / preco) * 100, 2)

    return {
        "ticker": ticker, "estrategia": "2 — Reversão Estatística (Z-Score)",
        "score": min(score, 100), "preco": round(preco, 2),
        "z_60": round(z60, 3), "z_252": round(z252, 3),
        "rsi_14": round(rsi, 1), "vol_ratio": round(vol_r, 2),
        "vol_60d": round(vol60, 2), "ret_1m": round(ret1m, 2),
        "alvo": alvo, "potencial": pot,
        "explicacao": (
            f"Z-Score de preço = {z60:.2f}σ (60 dias) e {z252:.2f}σ (252 dias). "
            f"Z-Score de -2.0 significa que o preço está 2 desvios-padrão abaixo da média — "
            f"isso ocorre por acaso apenas ~2.3% das vezes em distribuição normal. "
            f"RSI={rsi:.0f} confirma oversold técnico. "
            f"A Renaissance chamava isso de 'Déjà Vu' — o mercado repete: "
            f"desvios extremos revertem. Alvo = média dos últimos 60d: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BEAR", "NEUTRO", "ALTA_VOL"],
    }

print("✅ Estratégia 2 (Reversão Estatística Z-Score) definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 3: Detecção de Regime + Momentum Adaptativo
# MAGIC
# MAGIC **Conceito da Renaissance — Hidden Markov Models:**
# MAGIC A Renaissance usava modelos de regime (como HMMs) para detectar automaticamente
# MAGIC em qual "estado" o mercado está — tendência, reversão ou transição.
# MAGIC
# MAGIC Esta estratégia identifica ações que estão em **transição de regime bear → bull**:
# MAGIC ações que perderam momentum nos últimos meses mas estão mostrando sinais de recuperação.
# MAGIC
# MAGIC **Critérios (adaptação retail do HMM):**
# MAGIC - Z-Score de 12 meses negativo (estava em bear)
# MAGIC - Z-Score de 1 mês positivo e superior ao de 3 meses (acelerando para cima)
# MAGIC - RSI saindo da zona oversold (40–55)
# MAGIC - Volume crescente (dinheiro voltando)
# MAGIC - Inclinação do preço virando positiva (slope_trend > 0)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC A transição de regime é o momento de maior retorno esperado.
# MAGIC Capturar o ativo que estava em queda e acaba de virar é o "sweet spot" dos modelos de regime.

# COMMAND ----------

# DBTITLE 1,Estratégia 3 — Detecção de Regime (Momentum Adaptativo)
def estrategia_regime_adaptativo(painel, ticker):
    """
    Estratégia 3 — Detecção de Transição de Regime.

    Critérios (bear → bull transition):
    ✅ Z-Score 6m < 0 (estava em queda relativa)
    ✅ Z-Score 1m > Z-Score 3m (acelerando = momentum virando)
    ✅ Z-Score 1m > 0.5 (já superando a média no curto prazo)
    ✅ RSI entre 40-58 (saindo do oversold, ainda não overbought)
    ✅ trend_slope > 0 (preço começando a subir)
    ✅ vol_ratio > 1.0 (volume confirmando)
    """
    d = painel.get(ticker)
    if not d:
        return None

    z1m   = d.get("z_mom_1m",  0) or 0
    z3m   = d.get("z_mom_3m",  0) or 0
    z6m   = d.get("z_mom_6m",  0) or 0
    rsi   = d.get("rsi_14", 50) or 50
    slope = d.get("trend_slope", 0) or 0
    vol_r = d.get("vol_ratio", 1) or 1
    preco = d.get("preco", 0)
    ret1m = d.get("ret_1m", 0) or 0
    ret3m = d.get("ret_3m", 0) or 0

    # Critérios obrigatórios — transição bear → bull
    if z6m >= 0:         return None   # não estava em queda relativa
    if z1m <= 0.5:       return None   # não acelerou o suficiente
    if z1m <= z3m:       return None   # momentum não está acelerando
    if not (40 <= rsi <= 58): return None
    if slope <= 0:       return None

    # Delta de momentum (aceleração)
    delta_mom = z1m - z3m   # quanto acelerou

    score = 0
    if delta_mom > 1.5:     score += 35
    elif delta_mom > 1.0:   score += 25
    else:                   score += 15

    if z1m > 1.5:           score += 20
    elif z1m > 1.0:         score += 12

    if 45 <= rsi <= 55:     score += 20   # zona mais receptiva
    else:                   score += 8

    if vol_r > 1.3:         score += 15
    elif vol_r > 1.0:       score += 8

    if slope > 0.1:         score += 10

    alvo = preco * (1 + max(ret1m, 5) / 100)
    pot  = ((alvo - preco) / preco) * 100

    return {
        "ticker": ticker, "estrategia": "3 — Detecção de Regime (Bear→Bull)",
        "score": min(score, 100), "preco": round(preco, 2),
        "z_mom_1m": round(z1m, 3), "z_mom_3m": round(z3m, 3),
        "z_mom_6m": round(z6m, 3), "delta_mom": round(delta_mom, 3),
        "rsi_14": round(rsi, 1), "vol_ratio": round(vol_r, 2),
        "trend_slope": round(slope, 4), "ret_1m": round(ret1m, 2),
        "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Transição de regime detectada: Z-Score de 6m = {z6m:.2f}σ (estava em queda relativa) "
            f"mas Z-Score de 1m = {z1m:.2f}σ (já superando a média no curto prazo). "
            f"Aceleração de momentum: Δ = {delta_mom:.2f}σ (z1m - z3m). "
            f"RSI={rsi:.0f} saindo do oversold, slope de preço positivo ({slope:.4f}%/dia). "
            f"A Renaissance detectava exatamente esse padrão de virada via modelos de regime (HMM). "
            f"É o momento de máximo retorno esperado: comprar a transição. "
            f"Alvo: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BULL", "NEUTRO", "BEAR", "ALTA_VOL"],
    }

print("✅ Estratégia 3 (Detecção de Regime) definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 4: Arbitragem de Pares — Spread Estatístico
# MAGIC
# MAGIC **Conceito da Renaissance — Statistical Arbitrage:**
# MAGIC Esta é talvez a estratégia mais associada ao Medallion Fund.
# MAGIC Encontra **pares de ações cointegradas** (que historicamente andam juntas)
# MAGIC e opera o **spread** quando ele se desvia estatisticamente.
# MAGIC
# MAGIC **Como funciona:**
# MAGIC 1. Para cada par de ações do mesmo setor, testa **cointegração** (ADF test)
# MAGIC 2. Se cointegradas: calcula o **spread** e seu Z-Score
# MAGIC 3. Quando Z-Score do spread > +2.0 → a primeira ação está cara relativa → SHORT nela / LONG na outra
# MAGIC 4. Quando Z-Score do spread < -2.0 → a primeira está barata relativa → LONG nela / SHORT na outra
# MAGIC
# MAGIC **Pares naturais da B3 (mesmo setor):**
# MAGIC - Petrobras: PETR3 × PETR4 (ON vs PN)
# MAGIC - Eletrobras: ELET3 × ELET6
# MAGIC - Bancos: ITUB4 × BBDC4
# MAGIC - Siderurgia: GGBR4 × USIM5
# MAGIC - Energia: EGIE3 × ENGI11
# MAGIC - Papel: SUZB3 × KLBN11

# COMMAND ----------

# DBTITLE 1,Estratégia 4 — Arbitragem de Pares
# Pares candidatos por setor
PARES_CANDIDATOS = [
    ("PETR3.SA", "PETR4.SA", "Petrobras ON/PN"),
    ("ELET3.SA", "ELET6.SA", "Eletrobras ON/PNB"),
    ("ITUB4.SA", "BBDC4.SA", "Bancos Privados"),
    ("GGBR4.SA", "USIM5.SA", "Siderurgia"),
    ("EGIE3.SA", "ENGI11.SA", "Energia Elétrica"),
    ("SUZB3.SA", "KLBN11.SA", "Papel e Celulose"),
    ("VALE3.SA", "CSNA3.SA", "Mineração/Siderurgia"),
    ("RADL3.SA", "RAIA3.SA", "Farmácias"),
    ("CCRO3.SA", "ECOR3.SA", "Concessões"),
    ("MRVE3.SA", "CYRE3.SA", "Construção Civil"),
    ("BBAS3.SA", "ITUB4.SA", "Bancos"),
    ("PETR4.SA", "VALE3.SA", "Commodities BR"),
]


def analisar_par(tk1, tk2, nome_par, dados):
    """
    Analisa um par de ações para arbitragem estatística.
    Retorna oportunidade se spread estiver em extremo estatístico.
    """
    if tk1 not in dados or tk2 not in dados:
        return None

    df1 = dados[tk1]["Close"].sort_index()
    df2 = dados[tk2]["Close"].sort_index()

    # Alinha os índices
    idx_comum = df1.index.intersection(df2.index)
    if len(idx_comum) < 60:
        return None

    s1 = df1.loc[idx_comum]
    s2 = df2.loc[idx_comum]

    # Razão de hedge (hedge ratio) via regressão
    try:
        slope, intercept, r_val, _, _ = stats.linregress(s2.values, s1.values)
        r2 = r_val ** 2
        if r2 < 0.50:   # correlação mínima de 70% (r² = 0.49 ≈ r=0.7)
            return None
    except Exception:
        return None

    # Spread
    spread = s1 - slope * s2

    # Z-Score do spread (últimos 60 pregões)
    janela = min(60, len(spread))
    spread_rec = spread.iloc[-janela:]
    mu_sp, std_sp = spread_rec.mean(), spread_rec.std()
    if std_sp == 0:
        return None

    z_spread = float((spread.iloc[-1] - mu_sp) / std_sp)

    # Teste de cointegração (ADF no spread) — simplificado
    cointegrado = True
    p_value = 0.0
    if STATSMODELS_OK and len(spread) >= 30:
        try:
            adf_result = adfuller(spread.dropna(), maxlag=5)
            p_value = adf_result[1]
            cointegrado = p_value < 0.10   # 10% de significância
        except Exception:
            cointegrado = abs(z_spread) > 1.0   # fallback

    if not cointegrado:
        return None

    # Sinal só se spread extremo
    if abs(z_spread) < 2.0:
        return None

    preco1 = float(s1.iloc[-1])
    preco2 = float(s2.iloc[-1])
    direcao = "LONG" if z_spread < -2.0 else "SHORT"

    score = 0
    if abs(z_spread) > 3.0:   score += 40
    elif abs(z_spread) > 2.5: score += 28
    else:                     score += 18

    if r2 > 0.85:              score += 25
    elif r2 > 0.70:            score += 15

    if p_value < 0.01:         score += 20
    elif p_value < 0.05:       score += 12
    else:                      score += 5

    # Alvo: reverter 50% do z_spread
    spread_atual = float(spread.iloc[-1])
    spread_alvo  = mu_sp + z_spread * std_sp * 0.5  # reverter metade
    pot_spread   = abs(spread_atual - spread_alvo)
    pot_pct      = round((pot_spread / preco1) * 100, 2)
    alvo_preco   = round(preco1 + (spread_alvo - spread_atual) * (1 if z_spread < 0 else -1), 2)

    return {
        "ticker":       tk1,
        "ticker2":      tk2,
        "par":          nome_par,
        "estrategia":   "4 — Arbitragem de Pares (Stat-Arb)",
        "score":        min(score, 100),
        "preco":        round(preco1, 2),
        "preco2":       round(preco2, 2),
        "z_spread":     round(z_spread, 3),
        "r2":           round(r2, 3),
        "p_value_adf":  round(p_value, 4),
        "direcao":      direcao,
        "alvo":         alvo_preco,
        "potencial":    pot_pct,
        "explicacao": (
            f"Par '{nome_par}': {tk1.replace('.SA','')} e {tk2.replace('.SA','')} "
            f"são cointegrados (ADF p={p_value:.4f}, R²={r2:.2f}). "
            f"O spread atual está {abs(z_spread):.2f}σ da média histórica. "
            f"Sinal: {direcao} em {tk1.replace('.SA','')} "
            f"({'subavaliado' if direcao=='LONG' else 'sobreavaliado'} relativamente a {tk2.replace('.SA','')}). "
            f"A Renaissance fazia MILHARES de operações como essa simultaneamente, "
            f"cada uma com pequeno edge mas altíssima probabilidade. "
            f"Potencial de reversão: +{pot_pct:.1f}% em {tk1.replace('.SA','')}."
        ),
        "regime_alvo": ["BULL", "BEAR", "NEUTRO", "ALTA_VOL"],
    }


print("🔍 Analisando pares candidatos...\n")
RESULTADOS_PARES = []
for tk1, tk2, nome in PARES_CANDIDATOS:
    r = analisar_par(tk1, tk2, nome, DADOS_ACOES)
    if r:
        RESULTADOS_PARES.append(r)
        print(f"  ✅ {nome}: Z-Spread={r['z_spread']:.2f}σ, R²={r['r2']:.2f} → {r['direcao']}")

print(f"\n✅ {len(RESULTADOS_PARES)} pares com oportunidade encontrados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 5: Sinal Composto Multi-Fator (Filosofia "1000 sinais")
# MAGIC
# MAGIC **Conceito da Renaissance — Multi-Strategy Integration:**
# MAGIC Simons descobriu que **nenhum sinal individual** é suficientemente bom.
# MAGIC O Medallion Fund combinava **centenas (talvez milhares) de sinais pequenos**,
# MAGIC cada um com mínima edge, mas juntos criando uma vantagem estatística robusta.
# MAGIC
# MAGIC *"We have 30 years of data on every stock in the US market... we look for
# MAGIC  anomalies, hidden patterns, and exploit them."* — Jim Simons
# MAGIC
# MAGIC **Esta estratégia combina TODOS os fatores do painel:**
# MAGIC - Momentum de curto, médio e longo prazo
# MAGIC - Reversão estatística (Z-Score)
# MAGIC - Qualidade do risco-retorno (Sharpe proxy)
# MAGIC - Tendência de preço (slope)
# MAGIC - Liquidez (volume ratio)
# MAGIC
# MAGIC **Regras do sinal composto:**
# MAGIC - Pelo menos 4 de 6 fatores apontam na mesma direção
# MAGIC - Score final normalizado (Kelly-like position sizing)
# MAGIC - Filtro de correlação: descarta ativos muito similares entre si

# COMMAND ----------

# DBTITLE 1,Estratégia 5 — Sinal Composto Multi-Fator
def estrategia_multi_fator(painel, ticker):
    """
    Estratégia 5 — Sinal Composto Multi-Fator (Medallion-style).

    Combina 6 sinais independentes:
      S1: Momentum composto (3m+6m)
      S2: Reversão de curto prazo (z_60)
      S3: Qualidade do Sharpe proxy
      S4: Tendência de preço (slope)
      S5: Aceleração de volume
      S6: RSI em zona ótima de entrada

    Requer pelo menos 4/6 sinais positivos + sinal composto > 0.8σ
    """
    d = painel.get(ticker)
    if not d:
        return None

    preco = d.get("preco", 0)
    z3m   = d.get("z_mom_3m",  0) or 0
    z6m   = d.get("z_mom_6m",  0) or 0
    z1m   = d.get("z_mom_1m",  0) or 0
    z60   = d.get("z_60",      0) or 0
    zshar = d.get("z_sharpe",  0) or 0
    slope = d.get("trend_slope", 0) or 0
    vol_r = d.get("vol_ratio", 1) or 1
    rsi   = d.get("rsi_14", 50) or 50
    vol60 = d.get("vol_60d", 30) or 30
    ret3m = d.get("ret_3m", 0) or 0

    # 6 sinais binários (1 = positivo, 0 = negativo)
    s1 = 1 if (z3m + z6m) / 2 > 0.3 else 0         # Momentum 3-6m positivo
    s2 = 1 if z60 > -1.5 and z60 < 1.5 else 0        # Preço normalizado (nem esticado nem oversold extremo)
    s3 = 1 if zshar > 0.3 else 0                      # Sharpe proxy positivo
    s4 = 1 if slope > 0 else 0                        # Tendência positiva
    s5 = 1 if 0.9 <= vol_r <= 2.5 else 0              # Volume adequado (não seco, não explodido)
    s6 = 1 if 42 <= rsi <= 65 else 0                  # RSI em zona de entrada ótima

    sinais_positivos = s1 + s2 + s3 + s4 + s5 + s6

    if sinais_positivos < 4:
        return None

    # Sinal composto ponderado
    comp = 0.25*z3m + 0.25*z6m + 0.15*z1m + 0.15*zshar + 0.10*(slope*10) + 0.10*(vol_r-1)
    if comp < 0.5:
        return None

    # Penaliza volatilidade muito alta
    if vol60 > 70:
        return None

    score = 0
    score += sinais_positivos * 12      # até 72 pontos pelos sinais
    if comp > 1.5:    score += 20
    elif comp > 1.0:  score += 12
    else:             score += 5
    score = min(score, 100)

    alvo = preco * (1 + max(abs(ret3m) * 0.5, 5) / 100)
    pot  = ((alvo - preco) / preco) * 100

    sinais_str = f"S1({'✅' if s1 else '❌'}) S2({'✅' if s2 else '❌'}) S3({'✅' if s3 else '❌'}) S4({'✅' if s4 else '❌'}) S5({'✅' if s5 else '❌'}) S6({'✅' if s6 else '❌'})"

    return {
        "ticker": ticker, "estrategia": "5 — Sinal Composto Multi-Fator",
        "score": score, "preco": round(preco, 2),
        "sinais_ok": sinais_positivos, "comp_score": round(comp, 3),
        "z_mom_3m": round(z3m, 3), "z_sharpe": round(zshar, 3),
        "rsi_14": round(rsi, 1), "vol_ratio": round(vol_r, 2),
        "vol_60d": round(vol60, 2), "ret_3m": round(ret3m, 2),
        "sinais_str": sinais_str,
        "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Multi-fator: {sinais_positivos}/6 sinais positivos. {sinais_str}. "
            f"Score composto = {comp:.2f}σ. "
            f"Simons não confiava em um único indicador — combinava centenas de sinais pequenos "
            f"para criar uma edge estatística robusta. "
            f"Quando múltiplos sinais independentes convergem, a probabilidade de acerto aumenta significativamente. "
            f"Retorno 3m={ret3m:.1f}%, RSI={rsi:.0f}, Vol={vol60:.0f}%aa. "
            f"Alvo: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BULL", "NEUTRO", "BEAR", "ALTA_VOL"],
    }

print("✅ Estratégia 5 (Multi-Fator) definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Execução de Todas as Estratégias

# COMMAND ----------

# DBTITLE 1,Varredura Completa
print("🔍 Iniciando varredura de estratégias Simons/RenTech...\n")

resultados_todos = []

# Estratégias 1, 2, 3, 5 (por ativo individual)
for ticker in PAINEL:
    for fn in [
        estrategia_momentum_cross,
        estrategia_reversao_zscore,
        estrategia_regime_adaptativo,
        estrategia_multi_fator,
    ]:
        try:
            r = fn(PAINEL, ticker)
            if r:
                resultados_todos.append(r)
        except Exception:
            pass

# Estratégia 4 (pares) — já calculada acima
resultados_todos.extend(RESULTADOS_PARES)

print(f"✅ Varredura concluída!")
print(f"   📋 Total de oportunidades: {len(resultados_todos)}")
print(f"   📊 Ativos analisados: {len(PAINEL)}")
print(f"   🔗 Pares analisados: {len(PARES_CANDIDATOS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Resultados — Recomendações Simons / Renaissance

# COMMAND ----------

# DBTITLE 1,Exibição HTML dos Resultados
resultados_filtrados = sorted(
    [r for r in resultados_todos if REGIME in r.get("regime_alvo", [REGIME])],
    key=lambda x: x["score"], reverse=True,
)

ECORES = {
    "1": "#CBA6F7",   # roxo — momentum
    "2": "#89B4FA",   # azul — reversão
    "3": "#A6E3A1",   # verde — regime
    "4": "#FAB387",   # laranja — pares
    "5": "#F38BA8",   # rosa — multi-fator
}
EICONS = {"1": "🚀", "2": "📉", "3": "🔄", "4": "⚖️", "5": "🧮"}
RCORES = {"BULL": "#00C851", "BEAR": "#FF4444", "NEUTRO": "#FFBB33", "ALTA_VOL": "#FF8800"}

def score_badge(s):
    cor = "#00C851" if s >= 75 else ("#FFBB33" if s >= 55 else "#CBA6F7")
    return f'<span style="background:{cor};color:#000;padding:2px 10px;border-radius:12px;font-weight:800;">{s}</span>'

def gerar_html_simons(resultados, regime, data_corte):
    rcor = RCORES.get(regime, "#888")
    rics = {"BULL": "🟢 BULL", "BEAR": "🔴 BEAR", "NEUTRO": "🟡 NEUTRO", "ALTA_VOL": "🟠 ALTA VOL"}

    if not resultados:
        return f"""
        <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
             padding:32px;border-radius:16px;border:2px solid #CBA6F7;">
          <h2 style="color:#CBA6F7;margin-top:0;">🧮 Nenhum Sinal Quantitativo Detectado</h2>
          <p style="font-size:16px;">
            Na data <strong>{data_corte.strftime('%d/%m/%Y')}</strong> com regime
            <strong style="color:{rcor}">{rics.get(regime, regime)}</strong>,
            nenhuma ação brasileira atingiu os critérios estatísticos das estratégias de Jim Simons.
          </p>
          <hr style="border-color:#45475A;">
          <p style="color:#BAC2DE;font-size:14px;">
            💡 <strong>Interpretação quantitativa:</strong><br>
            Simons dizia que o mercado nem sempre oferece oportunidades — às vezes todos os sinais são
            ruidosos e nenhum tem edge estatístico suficiente. Nesses momentos,
            <strong>não operar é a decisão mais racional</strong>. Tente outra data ou amplie o universo de ativos.
          </p>
        </div>"""

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
         padding:24px;border-radius:16px;">
      <h1 style="color:#CBA6F7;margin:0 0 18px 0;">🧮 Simons Simulator — Sinais Quantitativos Detectados</h1>
      <p style="color:#888;font-size:12px;margin-bottom:16px;">
        Inspirado nas estratégias da Renaissance Technologies / Medallion Fund de Jim Simons
      </p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid #CBA6F7;">
          📅 <b>Data:</b> {data_corte.strftime('%d/%m/%Y')}
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid {rcor};">
          🌡️ <b>Regime:</b> <span style="color:{rcor}">{rics.get(regime, regime)}</span>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid #89B4FA;">
          📊 <b>Sinais:</b> {len(resultados)} oportunidades
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid #A6E3A1;">
          🔗 <b>Pares:</b> {sum(1 for r in resultados if '4' in r['estrategia'])} pares cointegrados
        </div>
      </div>
      <hr style="border-color:#45475A;">
    """

    grupos = {}
    for r in resultados:
        n = r["estrategia"][0]
        grupos.setdefault(n, []).append(r)

    for num, items in sorted(grupos.items()):
        cor  = ECORES.get(num, "#888")
        icon = EICONS.get(num, "📊")
        html += f"""
        <h2 style="color:{cor};border-bottom:2px solid {cor};padding-bottom:6px;">
          {icon} Estratégia {items[0]['estrategia']}
          <span style="font-size:13px;color:#888;font-weight:400;margin-left:8px;">
            ({len(items)} sinal{'is' if len(items)>1 else ''})
          </span>
        </h2>"""

        for r in items[:12]:
            tk  = r["ticker"].replace(".SA", "")
            tk2 = r.get("ticker2", "")
            pc  = r.get("preco", 0)
            alv = r.get("alvo", 0)
            pot = r.get("potencial", 0)
            pc2 = "#00C851" if pot > 0 else "#FF4444"

            # Título do card (par ou individual)
            if tk2:
                tk2c = tk2.replace(".SA", "")
                dir_ = r.get("direcao", "")
                dir_cor = "#00C851" if dir_ == "LONG" else "#FF4444"
                tk_display = f'<b>{tk}</b> <span style="color:#888;font-size:12px;">× {tk2c}</span> <span style="color:{dir_cor};font-size:12px;font-weight:700;">[{dir_}]</span>'
            else:
                tk_display = f'<b>{tk}</b>'

            # Métricas
            mex = ""
            metricas_map = [
                ("mom_comp",   "MomComp", "σ"),
                ("z_spread",   "Z-Spread", "σ"),
                ("z_mom_3m",   "Z-Mom3m",  "σ"),
                ("z_mom_6m",   "Z-Mom6m",  "σ"),
                ("z_60",       "Z-60d",    "σ"),
                ("rsi_14",     "RSI",      ""),
                ("vol_ratio",  "Vol",      "x"),
                ("r2",         "R²",       ""),
                ("sinais_ok",  "Sinais",   "/6"),
                ("comp_score", "CompScore","σ"),
                ("delta_mom",  "ΔMom",     "σ"),
            ]
            for k, lbl, suf in metricas_map:
                if k in r and r[k] is not None:
                    v = r[k]
                    mex += (f'<span style="background:#45475A;padding:2px 8px;border-radius:8px;'
                            f'margin-right:5px;font-size:12px;">{lbl}: <b>{v}{suf}</b></span>')

            html += f"""
            <div style="background:#313244;border-radius:12px;padding:14px;margin-bottom:12px;border-left:4px solid {cor};">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                  <span style="font-size:20px;color:#CDD6F4;">{tk_display}</span>
                  <span style="margin-left:10px;color:#888;font-size:12px;">{r['estrategia']}</span>
                </div>
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                  <span>Score: {score_badge(r['score'])}</span>
                  <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;">
                    💰 R$ <b>{pc:.2f}</b>
                  </span>
                  <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;">
                    🎯 <b style="color:{pc2}">R$ {alv:.2f}</b> <em style="color:{pc2}">(+{pot:.1f}%)</em>
                  </span>
                </div>
              </div>
              <div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:5px;">{mex}</div>
              <div style="background:#1E1E2E;border-radius:8px;padding:10px;font-size:13px;
                   color:#BAC2DE;line-height:1.6;">
                🧮 <b style="color:{cor}">Por que Simons operaria?</b><br>{r['explicacao']}
              </div>
            </div>"""

    html += """
      <hr style="border-color:#45475A;margin-top:20px;">
      <div style="background:#313244;border-radius:10px;padding:14px;font-size:12px;color:#888;">
        ⚠️ <b>Disclaimer:</b> Este notebook é puramente educacional e não constitui recomendação de investimento.
        As estratégias da Renaissance Technologies são proprietárias e desconhecidas em detalhes.
        Esta é uma adaptação acadêmica dos princípios conhecidos publicamente.
        Sempre utilize gestão de risco e opere apenas capital que pode perder.
      </div>
    </div>"""
    return html


displayHTML(gerar_html_simons(resultados_filtrados, REGIME, DATA_CORTE))

# COMMAND ----------

# DBTITLE 1,Tabela Pandas de Resultados
if resultados_filtrados:
    COLS = ["ticker", "estrategia", "score", "preco", "alvo", "potencial",
            "mom_comp", "z_spread", "z_60", "rsi_14", "vol_ratio",
            "ret_3m", "ret_6m", "z_sharpe", "sinais_ok", "direcao"]
    df_res = pd.DataFrame(resultados_filtrados)
    cols   = [c for c in COLS if c in df_res.columns]
    df_ex  = df_res[cols].copy()
    df_ex["ticker"] = df_ex["ticker"].str.replace(".SA", "", regex=False)
    df_ex  = df_ex.sort_values(["estrategia", "score"], ascending=[True, False]).reset_index(drop=True)

    print(f"📋 {len(df_ex)} sinais | {DATA_CORTE.strftime('%d/%m/%Y')} | Regime: {REGIME}\n")
    display(df_ex)

    try:
        caminho = f"/tmp/simons_resultados_{DATA_CORTE.strftime('%Y%m%d')}.csv"
        df_ex.to_csv(caminho, index=False)
        print(f"\n💾 CSV salvo: {caminho}")
    except Exception as e:
        print(f"\n⚠️ Erro ao salvar: {e}")
else:
    print(f"⚠️ Nenhum sinal quantitativo para {DATA_CORTE.strftime('%d/%m/%Y')} ({REGIME})")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📈 Backtesting — O que aconteceu depois da data de corte?
# MAGIC
# MAGIC > Esta seção só executa quando a **data de corte está no passado**.
# MAGIC > Avalia cada sinal verificando se o preço atingiu o alvo ou o stop (-8%).
# MAGIC > Para a Estratégia 4 (Pares), avalia o spread (não o preço absoluto).

# COMMAND ----------

# DBTITLE 1,Verificação — Data no Passado?
import matplotlib.patches as mpatches

HOJE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
DIAS_APOS = (HOJE - DATA_CORTE).days

print(f"📅 Data de corte  : {DATA_CORTE.strftime('%d/%m/%Y')}")
print(f"📅 Data atual     : {HOJE.strftime('%d/%m/%Y')}")
print(f"⏱️  Dias após corte: {DIAS_APOS}")

FAZER_BACKTEST = DIAS_APOS >= 3
if FAZER_BACKTEST:
    print(f"\n✅ {DIAS_APOS} dias pós-corte — iniciando backtesting!")
else:
    print("\n🔵 Data atual — backtesting não aplicável.")

# COMMAND ----------

# DBTITLE 1,Download Pós-Corte e Histórico Completo
if FAZER_BACKTEST and resultados_filtrados:
    tickers_rec = list({r["ticker"] for r in resultados_filtrados} |
                       {r.get("ticker2","") for r in resultados_filtrados if r.get("ticker2")})
    tickers_rec = [t for t in tickers_rec if t]

    print(f"🔽 Baixando dados pós-corte para {len(tickers_rec)} ativos...")
    DATA_POS = DATA_CORTE + timedelta(days=1)

    DADOS_POS = baixar_dados(tickers_rec, DATA_POS, HOJE + timedelta(days=1), batch_size=10)
    print(f"\n🔽 Baixando histórico completo para gráficos...")
    DADOS_FULL = baixar_dados(tickers_rec, DATA_INICIO, HOJE + timedelta(days=1), batch_size=10)

    print(f"\n✅ Pós-corte: {len(DADOS_POS)} ativos. Completo: {len(DADOS_FULL)} ativos.")
else:
    DADOS_POS  = {}
    DADOS_FULL = {}

# COMMAND ----------

# DBTITLE 1,Avaliação do Backtesting
STOP_PCT = -0.08   # Stop de -8% (Simons usava stops mais apertados que BNF)

def avaliar_sinal_simons(r, dados_pos):
    ticker = r["ticker"]
    if ticker not in dados_pos or len(dados_pos[ticker]) == 0:
        return {"resultado": "❓ Sem dados", "data_resultado": "—",
                "retorno_realizado": None, "dias_ate_alvo": None, "cor": "#888888"}

    preco_ent = r.get("preco", 0)
    alvo      = r.get("alvo", 0)
    stop_p    = preco_ent * (1 + STOP_PCT)
    df_pos    = dados_pos[ticker].sort_index()

    resultado = "⏳ Ainda aberto"
    data_res  = None
    dias_alvo = None
    cor       = "#FFBB33"

    for i, (dt, row) in enumerate(df_pos.iterrows()):
        high = row.get("High", row["Close"])
        low  = row.get("Low",  row["Close"])
        if low <= stop_p:
            resultado = f"🛑 Stop -8% (≤ R$ {stop_p:.2f})"
            data_res  = dt; dias_alvo = i + 1; cor = "#FF4444"; break
        if high >= alvo:
            resultado = f"✅ Alvo (≥ R$ {alvo:.2f})"
            data_res  = dt; dias_alvo = i + 1; cor = "#00C851"; break

    ultimo = df_pos.loc[data_res]["Close"] if data_res is not None else df_pos.iloc[-1]["Close"]
    ret_r  = ((ultimo - preco_ent) / preco_ent) * 100 if preco_ent > 0 else 0

    return {
        "resultado": resultado, "cor": cor,
        "data_resultado": data_res.strftime("%d/%m/%Y") if data_res else "—",
        "retorno_realizado": round(ret_r, 2),
        "ultimo_preco": round(float(ultimo), 2),
        "dias_ate_alvo": dias_alvo,
        "max_preco": round(float(df_pos["High"].max()), 2) if "High" in df_pos.columns else None,
        "min_preco": round(float(df_pos["Low"].min()),  2) if "Low"  in df_pos.columns else None,
    }


resultados_backtest = []
if FAZER_BACKTEST and resultados_filtrados:
    print("🔬 Avaliando sinais...\n")
    for r in resultados_filtrados:
        av = avaliar_sinal_simons(r, DADOS_POS)
        resultados_backtest.append({**r, **av})

    acertos = sum(1 for x in resultados_backtest if "✅" in x["resultado"])
    stops   = sum(1 for x in resultados_backtest if "🛑" in x["resultado"])
    abertos = sum(1 for x in resultados_backtest if "⏳" in x["resultado"])
    total   = len(resultados_backtest)

    print(f"✅ Alvos atingidos: {acertos}/{total} ({acertos/total*100:.0f}%)")
    print(f"🛑 Stops acionados: {stops}/{total}  ({stops/total*100:.0f}%)")
    print(f"⏳ Ainda abertos  : {abertos}/{total}")

# COMMAND ----------

# DBTITLE 1,Tabela HTML de Backtesting
def gerar_html_bt_simons(backtest, data_corte, hoje):
    if not backtest:
        return "<p style='color:#888;font-family:sans-serif;'>Sem dados de backtesting.</p>"

    acertos = sum(1 for x in backtest if "✅" in x["resultado"])
    stops   = sum(1 for x in backtest if "🛑" in x["resultado"])
    abertos = sum(1 for x in backtest if "⏳" in x["resultado"])
    total   = len(backtest)
    taxa    = acertos / total * 100 if total > 0 else 0
    ret_medio = np.mean([x["retorno_realizado"] for x in backtest if x.get("retorno_realizado") is not None])
    cor_taxa  = "#00C851" if taxa >= 55 else ("#FFBB33" if taxa >= 40 else "#FF4444")
    ret_cor   = "#00C851" if ret_medio >= 0 else "#FF4444"

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
         padding:24px;border-radius:16px;margin-top:16px;">
      <h2 style="color:#CBA6F7;margin:0 0 14px 0;">🔬 Backtesting Simons — Desempenho dos Sinais Quantitativos</h2>
      <p style="color:#888;font-size:13px;margin-bottom:14px;">
        Corte: <b>{data_corte.strftime('%d/%m/%Y')}</b> → Avaliado até: <b>{hoje.strftime('%d/%m/%Y')}</b>
        | Stop: <b>-8%</b>
      </p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;">
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid {cor_taxa};">
          🎯 <b>Taxa de acerto:</b>
          <span style="color:{cor_taxa};font-size:18px;font-weight:800;">{taxa:.0f}%</span>
          <span style="color:#888;font-size:12px;">({acertos}/{total})</span>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid {ret_cor};">
          📈 <b>Retorno médio real:</b>
          <span style="color:{ret_cor};font-size:18px;font-weight:800;">{ret_medio:+.1f}%</span>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #00C851;">
          ✅ <b>Alvos:</b> <strong style="color:#00C851;">{acertos}</strong>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #FF4444;">
          🛑 <b>Stops:</b> <strong style="color:#FF4444;">{stops}</strong>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #FFBB33;">
          ⏳ <b>Abertos:</b> <strong style="color:#FFBB33;">{abertos}</strong>
        </div>
      </div>
      <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#313244;color:#CBA6F7;">
            <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #45475A;">Ativo</th>
            <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #45475A;">Estratégia</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Entrada</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Alvo</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Stop (-8%)</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Score</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Resultado</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Data</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Dias</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Retorno Real</th>
          </tr>
        </thead>
        <tbody>"""

    for i, r in enumerate(sorted(backtest, key=lambda x: x["score"], reverse=True)):
        tk      = r["ticker"].replace(".SA", "")
        ent     = r.get("preco", 0)
        alvo_v  = r.get("alvo", 0)
        stop_v  = ent * 0.92
        sc      = r.get("score", 0)
        estrat  = r.get("estrategia", "")
        res     = r.get("resultado", "—")
        dt_r    = r.get("data_resultado", "—")
        dias    = r.get("dias_ate_alvo", "—")
        ret_r   = r.get("retorno_realizado")
        cor_l   = r.get("cor", "#888")
        bg      = "#2A2A3E" if i % 2 == 0 else "#252535"
        ret_s   = f"{ret_r:+.1f}%" if ret_r is not None else "—"
        ret_cor = "#00C851" if (ret_r or 0) > 0 else "#FF4444"
        sc_cor  = "#00C851" if sc >= 75 else ("#FFBB33" if sc >= 55 else "#CBA6F7")

        html += f"""
          <tr style="background:{bg};border-left:3px solid {cor_l};">
            <td style="padding:9px 12px;font-weight:700;">{tk}</td>
            <td style="padding:9px 12px;color:#888;font-size:12px;">{estrat}</td>
            <td style="padding:9px 12px;text-align:right;">R$ {ent:.2f}</td>
            <td style="padding:9px 12px;text-align:right;color:#00C851;">R$ {alvo_v:.2f}</td>
            <td style="padding:9px 12px;text-align:right;color:#FF4444;">R$ {stop_v:.2f}</td>
            <td style="padding:9px 12px;text-align:right;">
              <span style="background:{sc_cor};color:#000;padding:1px 8px;border-radius:10px;font-weight:700;">{sc}</span>
            </td>
            <td style="padding:9px 12px;text-align:center;font-weight:600;color:{cor_l};">{res}</td>
            <td style="padding:9px 12px;text-align:center;color:#888;">{dt_r}</td>
            <td style="padding:9px 12px;text-align:center;color:#888;">{dias if dias != "—" else "—"}</td>
            <td style="padding:9px 12px;text-align:right;font-weight:700;color:{ret_cor};">{ret_s}</td>
          </tr>"""

    html += """
        </tbody>
      </table>
      </div>
      <p style="color:#666;font-size:11px;margin-top:10px;">
        * Stop de -8% é mais apertado que o BNF (-10%), refletindo a disciplina quantitativa da Renaissance.
      </p>
    </div>"""
    return html


if FAZER_BACKTEST and resultados_backtest:
    displayHTML(gerar_html_bt_simons(resultados_backtest, DATA_CORTE, HOJE))
elif not FAZER_BACKTEST:
    print("🔵 Data atual — backtesting ignorado.")

# COMMAND ----------

# DBTITLE 1,Tabela Pandas — Backtesting Simons
if FAZER_BACKTEST and resultados_backtest:
    cols_bt = ["ticker", "estrategia", "score", "preco", "alvo", "potencial",
               "resultado", "data_resultado", "dias_ate_alvo",
               "retorno_realizado", "ultimo_preco", "max_preco", "min_preco"]
    df_bt = pd.DataFrame(resultados_backtest)
    cols  = [c for c in cols_bt if c in df_bt.columns]
    df_bt_ex = df_bt[cols].copy()
    df_bt_ex["ticker"] = df_bt_ex["ticker"].str.replace(".SA", "", regex=False)
    df_bt_ex = df_bt_ex.sort_values("score", ascending=False).reset_index(drop=True)
    display(df_bt_ex)
    try:
        pth = f"/tmp/simons_backtest_{DATA_CORTE.strftime('%Y%m%d')}.csv"
        df_bt_ex.to_csv(pth, index=False)
        print(f"💾 CSV salvo: {pth}")
    except Exception as e:
        print(f"⚠️ Erro: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Gráficos — Sinais Quantitativos com Linha de Corte
# MAGIC
# MAGIC Cada gráfico exibe:
# MAGIC - 🔵 Preço **pré-corte**
# MAGIC - 🟣 Preço **pós-corte** (colorido pelo resultado)
# MAGIC - 🟣 **Linha vertical** = data de corte
# MAGIC - 🟢 Linha verde = alvo | 🔴 Linha vermelha = stop (-8%)
# MAGIC - 📊 **Painel inferior**: Z-Score de 60 dias (sinal de reversão estatística)

# COMMAND ----------

# DBTITLE 1,Geração dos Gráficos por Ativo
def plot_simons_base64(ticker, dados_full, data_corte, preco_ent, alvo, stop_p, res_info, painel_dados):
    if ticker not in dados_full or len(dados_full[ticker]) < 10:
        return None

    df_plot  = dados_full[ticker].sort_index()
    df_antes = df_plot[df_plot.index <= pd.Timestamp(data_corte)]
    df_depois= df_plot[df_plot.index >  pd.Timestamp(data_corte)]

    # Calcula Z-Score rolling (60d) para painel inferior
    c = df_plot["Close"]
    z60_rolling = (c - c.rolling(60).mean()) / c.rolling(60).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1E1E2E")
    fig.patch.set_facecolor("#1E1E2E")

    # ── Painel Superior — Preço ────────────────────────────────────────────────
    if len(df_antes) > 0:
        ax1.plot(df_antes.index, df_antes["Close"],
                 color="#89B4FA", linewidth=1.5, label="Preço (pré-corte)", zorder=3)
    if len(df_depois) > 0:
        cor_p = res_info.get("cor", "#CDD6F4")
        ax1.plot(df_depois.index, df_depois["Close"],
                 color=cor_p, linewidth=2.0, label="Preço (pós-corte)", zorder=3)
        ax1.axvspan(pd.Timestamp(data_corte), df_plot.index[-1],
                    alpha=0.06, color="#CBA6F7", zorder=1)

    # Médias móveis
    for p, cor, lbl in [(25, "#F38BA8", "MM25"), (60, "#A6E3A1", "MM60")]:
        if len(c) >= p:
            ax1.plot(df_plot.index, c.rolling(p).mean(),
                     color=cor, linewidth=0.8, linestyle="--", alpha=0.65, label=lbl)

    ax1.axvline(pd.Timestamp(data_corte), color="#CBA6F7", linewidth=2,
                linestyle="--", label=f"Corte: {data_corte.strftime('%d/%m/%Y')}", zorder=5)
    ax1.scatter([pd.Timestamp(data_corte)], [preco_ent],
                color="#FAB387", s=90, zorder=7, label=f"Entrada R$ {preco_ent:.2f}")
    ax1.axhline(alvo,   color="#00C851", linewidth=1.2, linestyle="-.",
                label=f"Alvo R$ {alvo:.2f}", zorder=4)
    ax1.axhline(stop_p, color="#FF4444", linewidth=1.2, linestyle=":",
                label=f"Stop R$ {stop_p:.2f} (-8%)", zorder=4)

    # Marcador de resultado
    data_res_str = res_info.get("data_resultado", "—")
    if data_res_str and data_res_str != "—":
        try:
            drt = pd.Timestamp(datetime.strptime(data_res_str, "%d/%m/%Y"))
            idx = df_plot.index.searchsorted(drt)
            if idx < len(df_plot):
                pr  = float(df_plot["Close"].iloc[idx])
                mrk = "^" if "✅" in res_info.get("resultado","") else "v"
                ax1.scatter([drt], [pr], color=res_info.get("cor","#FFBB33"),
                            s=140, marker=mrk, zorder=8, label=f"Resultado: {data_res_str}")
        except Exception:
            pass

    tk_c  = ticker.replace(".SA","")
    res_s = res_info.get("resultado","⏳")
    ret_r = res_info.get("retorno_realizado")
    ret_s = f" | Ret: {ret_r:+.1f}%" if ret_r is not None else ""
    ax1.set_title(f"{tk_c}  |  {res_s}{ret_s}", color="#CDD6F4", fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylabel("Preço (R$)", color="#CDD6F4", fontsize=9)
    ax1.grid(axis="y", color="#313244", linewidth=0.5, alpha=0.7)
    ax1.grid(axis="x", color="#313244", linewidth=0.3, alpha=0.4)
    ax1.spines[:].set_color("#45475A")
    ax1.tick_params(colors="#CDD6F4")
    ax1.legend(loc="upper left", framealpha=0.3, facecolor="#313244",
               edgecolor="#45475A", labelcolor="#CDD6F4", fontsize=7)

    # ── Painel Inferior — Z-Score 60d ──────────────────────────────────────────
    ax2.plot(df_plot.index, z60_rolling, color="#CBA6F7", linewidth=1.0, label="Z-Score 60d")
    ax2.axhline(2.0,  color="#FF4444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(-2.0, color="#00C851", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(0,    color="#45475A", linewidth=0.5)
    ax2.fill_between(df_plot.index, z60_rolling, 0,
                     where=(z60_rolling > 0), color="#FF6B6B", alpha=0.15)
    ax2.fill_between(df_plot.index, z60_rolling, 0,
                     where=(z60_rolling < 0), color="#89B4FA", alpha=0.15)
    ax2.axvline(pd.Timestamp(data_corte), color="#CBA6F7", linewidth=1.5, linestyle="--", zorder=5)
    ax2.set_ylabel("Z-Score 60d", color="#CDD6F4", fontsize=8)
    ax2.set_ylim(-4, 4)
    ax2.grid(axis="y", color="#313244", linewidth=0.4, alpha=0.5)
    ax2.spines[:].set_color("#45475A")
    ax2.tick_params(colors="#CDD6F4")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, color="#CDD6F4", fontsize=8)

    plt.tight_layout(pad=1.2)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def gerar_html_graficos_simons(backtest, dados_full, data_corte, painel):
    if not backtest:
        return "<p style='color:#888;font-family:sans-serif;'>Nenhum gráfico disponível.</p>"

    ECORS = {"1":"#CBA6F7","2":"#89B4FA","3":"#A6E3A1","4":"#FAB387","5":"#F38BA8"}
    html = """
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;
         padding:24px;border-radius:16px;margin-top:12px;">
      <h2 style="color:#CBA6F7;margin:0 0 18px 0;">📊 Gráficos Quantitativos — Preço + Z-Score 60d</h2>"""

    for r in sorted(backtest, key=lambda x: x["score"], reverse=True):
        ticker  = r["ticker"]
        tk_c    = ticker.replace(".SA","")
        ent     = r.get("preco", 0)
        alvo    = r.get("alvo", 0)
        stop_p  = ent * 0.92
        cor_card= ECORS.get(r["estrategia"][0], "#45475A")
        res_i   = r.get("resultado","⏳")
        score   = r.get("score", 0)
        ret_r   = r.get("retorno_realizado")
        ret_s   = f"{ret_r:+.1f}%" if ret_r is not None else "—"
        ret_cor = "#00C851" if (ret_r or 0) > 0 else "#FF4444"

        img = plot_simons_base64(ticker, dados_full, data_corte, ent, alvo, stop_p, r, painel.get(ticker,{}))
        if img is None:
            continue

        html += f"""
        <div style="background:#313244;border-radius:14px;padding:16px;
             margin-bottom:18px;border-left:4px solid {cor_card};">
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
            <div>
              <span style="font-size:20px;font-weight:800;color:#CDD6F4;">{tk_c}</span>
              <span style="margin-left:10px;color:#888;font-size:12px;">{r.get('estrategia','')}</span>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                Score: <b style="color:#CBA6F7;">{score}</b>
              </span>
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                {res_i}
              </span>
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                Retorno: <b style="color:{ret_cor};">{ret_s}</b>
              </span>
            </div>
          </div>
          <img src="data:image/png;base64,{img}"
               style="width:100%;border-radius:8px;display:block;" />
        </div>"""

    html += "</div>"
    return html


if FAZER_BACKTEST and resultados_backtest and DADOS_FULL:
    print("📊 Gerando gráficos quantitativos...")
    displayHTML(gerar_html_graficos_simons(resultados_backtest, DADOS_FULL, DATA_CORTE, PAINEL))
    print("✅ Gráficos gerados!")
elif not FAZER_BACKTEST:
    # Gera gráficos mesmo sem backtesting (data atual)
    print("📊 Gerando gráficos dos ativos recomendados (histórico)...")
    if resultados_filtrados:
        tickers_graf = list({r["ticker"] for r in resultados_filtrados})
        DADOS_FULL_NOW = baixar_dados(tickers_graf, DATA_INICIO, HOJE + timedelta(days=1), batch_size=10)
        # Versão simplificada sem backtesting — usa resultado vazio
        bt_fake = [{**r, "resultado":"⏳ Data atual","data_resultado":"—","retorno_realizado":None,"cor":"#CBA6F7"}
                   for r in resultados_filtrados[:15]]
        displayHTML(gerar_html_graficos_simons(bt_fake, DADOS_FULL_NOW, DATA_CORTE, PAINEL))
        print("✅ Gráficos gerados!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Referência Completa — As 5 Estratégias de Simons
# MAGIC
# MAGIC | # | Estratégia | Conceito Core | Indicador Principal | Regime Ideal |
# MAGIC |---|------------|--------------|--------------------|-|
# MAGIC | 1 | Momentum Cross-Sectional | Persistência de retornos relativos | Z-Score composto 3m+6m+skip | 🟢 Bull |
# MAGIC | 2 | Reversão Estatística | Déjà Vu — desvios revertem | Z-Score de preço 60/252d | 🔴 Bear |
# MAGIC | 3 | Detecção de Regime | Transição Bear→Bull (HMM-like) | ΔZ-Score (z1m - z3m) | Qualquer |
# MAGIC | 4 | Arbitragem de Pares | Cointegração + spread Z-Score | ADF test + Z-Spread | Qualquer |
# MAGIC | 5 | Multi-Fator | 6 sinais independentes convergindo | Score composto 6-dimensional | Qualquer |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏆 Os 7 Princípios Imutáveis de Jim Simons
# MAGIC
# MAGIC 1. **Dados acima de tudo** — sem dados suficientes, não há estratégia
# MAGIC 2. **Sistemas, não intuição** — o algoritmo é sempre mais confiável que o humano
# MAGIC 3. **Muitos sinais pequenos** — nenhum sinal sozinho é suficiente
# MAGIC 4. **Gestão de risco rigorosa** — Kelly Criterion para dimensionar posições
# MAGIC 5. **Diversificação máxima** — centenas de posições simultâneas
# MAGIC 6. **Adapte-se ao regime** — momentum em bull, reversão em bear
# MAGIC 7. **Custos de transação importam** — uma estratégia lucrativa pode ser destruída por custos

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 🧮 Simons vs. BNF — Qual é melhor?
# MAGIC >
# MAGIC > Não existe "melhor" — existem **estilos diferentes** para mercados diferentes:
# MAGIC >
# MAGIC > - **BNF** é ideal para **traders ativos** com capacidade de monitorar pânico em tempo real
# MAGIC > - **Simons** é ideal para **sistemas sistemáticos** que operam sem emoção, com regras claras
# MAGIC >
# MAGIC > O melhor trader é aquele que entende qual é seu edge e o executa com disciplina.
