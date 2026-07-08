# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# dependencies = [
#   "yfinance",
#   "requests",
#   "tqdm",
# ]
# ///
# MAGIC %md
# MAGIC # 🏆 BNF — Melhor Ação do Dia
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objetivo deste Notebook
# MAGIC
# MAGIC Ao contrário do **BNF Simulator** (que lista todas as oportunidades),
# MAGIC este notebook foca em um único objetivo: varrer toda a B3, aplicar as 5 estratégias
# MAGIC de Takashi Kotegawa (BNF) e selecionar o **único ativo com a melhor pontuação composta**.
# MAGIC Em seguida, gera um **laudo de análise completo** com 4 pilares:
# MAGIC
# MAGIC | Pilar | Indicadores |
# MAGIC |-------|------------|
# MAGIC | 🌊 **Volatilidade (BNF)** | Kairi Ritsu, ATR%, Bollinger %B/Width, Vol. Histórica |
# MAGIC | 📐 **Momentum & Tendência** | RSI 14, MACD, Estocástico, MM9/21/25/55/200 |
# MAGIC | 📦 **Volume** | Volume Ratio, OBV, Exaustão de Vendedores |
# MAGIC | 🏢 **Contexto de Mercado** | Regime IBOV (Bull/Bear/Neutro) |
# MAGIC
# MAGIC > **Metodologia**: Cada estratégia produz um score (0–100). O ativo com **maior score**
# MAGIC > no regime atual é declarado a **melhor compra do dia**, e recebe uma simulação de R$1.000.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Dependências

# COMMAND ----------

# DBTITLE 1,Instalação
# MAGIC %pip install yfinance requests tqdm

# COMMAND ----------

# DBTITLE 1,Imports
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO
from datetime import datetime, timedelta
from IPython.display import display, HTML
import time

warnings.filterwarnings("ignore")
print(f"✅ Imports OK | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📅 Data de Corte

# COMMAND ----------

# DBTITLE 1,Widget
dbutils.widgets.removeAll()
dbutils.widgets.text("data_corte", datetime.now().strftime("%Y-%m-%d"), "📅 Data de Corte")

# COMMAND ----------

# DBTITLE 1,Validação
RAW_DATA = dbutils.widgets.get("data_corte").strip()
try:
    DATA_CORTE = datetime.strptime(RAW_DATA, "%Y-%m-%d")
    DATA_INICIO = DATA_CORTE - timedelta(days=450)
    if DATA_CORTE > datetime.now():
        raise ValueError("Data futura!")
    print(f"✅ Corte: {DATA_CORTE.strftime('%d/%m/%Y')} | Início: {DATA_INICIO.strftime('%d/%m/%Y')}")
except ValueError as e:
    raise ValueError(f"❌ {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Tickers B3

# COMMAND ----------

# DBTITLE 1,Lista de Ativos
TICKERS_B3 = [
    "PETR4.SA","PETR3.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","BBAS3.SA",
    "ABEV3.SA","WEGE3.SA","B3SA3.SA","RENT3.SA","LREN3.SA","MGLU3.SA",
    "EMBR3.SA","JBSS3.SA","SUZB3.SA","KLBN11.SA","RAIL3.SA",
    "CSNA3.SA","USIM5.SA","GGBR4.SA","CSAN3.SA","EQTL3.SA","ELET3.SA",
    "ELET6.SA","CMIG4.SA","CPFE3.SA","CPLE6.SA","EGIE3.SA","ENGI11.SA",
    "TAEE11.SA","VIVT3.SA","TIMS3.SA","TOTS3.SA","PRIO3.SA","CYRE3.SA",
    "MRVE3.SA","TEND3.SA","EZTC3.SA","ALPA4.SA","SOMA3.SA",
    "ARZZ3.SA","NTCO3.SA","COGN3.SA","YDUQ3.SA","ANIM3.SA",
    "SANB11.SA","ITSA4.SA","BBSE3.SA","IRBR3.SA","PSSA3.SA","CIEL3.SA",
    "BRSR6.SA","BPAN4.SA","SULA11.SA",
    "VIIA3.SA","AMER3.SA","VVAR3.SA","GRND3.SA","SBFG3.SA","PETZ3.SA",
    "RADL3.SA","RAIA3.SA","FLRY3.SA","HAPV3.SA","QUAL3.SA","HYPE3.SA","DASA3.SA",
    "ENEV3.SA","NEOE3.SA","AURE3.SA","ALUP11.SA","CESP6.SA","ENBR3.SA","TRPL4.SA",
    "LWSA3.SA","CASH3.SA",
    "SLCE3.SA","SMTO3.SA","AGRO3.SA","BEEF3.SA","MRFG3.SA","BRFS3.SA","CAML3.SA","RAIZ4.SA",
    "MULT3.SA","ALSO3.SA","JHSF3.SA",
    "CCRO3.SA","ECOR3.SA","RDOR3.SA","LOGN3.SA","POMO4.SA","AZUL4.SA","GOLL4.SA",
    "DTEX3.SA","RANI3.SA",
    "^BVSP",
]
print(f"📋 {len(TICKERS_B3)-1} ações + IBOV")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔽 Download e Indicadores

# COMMAND ----------

# DBTITLE 1,Funções Core
def baixar_dados(tickers, inicio, fim, batch_size=15):
    dados = {}
    fim_s = fim.strftime("%Y-%m-%d"); ini_s = inicio.strftime("%Y-%m-%d")
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for idx, batch in enumerate(batches):
        print(f"  📦 Lote {idx+1}/{len(batches)}: {len(batch)} ativos...")
        try:
            raw = yf.download(batch, start=ini_s, end=fim_s, auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df = raw.xs(tk, axis=1, level=1).copy()
                        df.dropna(subset=["Close"], inplace=True)
                        if len(df) >= 40: dados[tk] = df
                    except Exception: pass
            else:
                tk = batch[0]; raw.dropna(subset=["Close"], inplace=True)
                if len(raw) >= 40: dados[tk] = raw
        except Exception as e: print(f"    ⚠️ {e}")
        time.sleep(0.4)
    return dados


def calcular_indicadores(df):
    d = df.copy()
    # MMs
    for p, name in [(9,"MM9"),(21,"MM21"),(25,"MM25"),(55,"MM55"),(200,"MM200")]:
        d[name] = d["Close"].rolling(p).mean()
    # Kairi
    d["KAIRI_25"] = ((d["Close"] - d["MM25"]) / d["MM25"]) * 100
    # RSI 14
    delta = d["Close"].diff()
    g = delta.clip(lower=0); pe = (-delta).clip(lower=0)
    rg = g.ewm(com=13, adjust=False).mean(); rl = pe.ewm(com=13, adjust=False).mean()
    d["RSI_14"] = 100 - (100 / (1 + rg / rl.replace(0, np.nan)))
    # ATR
    hl = d["High"]-d["Low"]; hcp = (d["High"]-d["Close"].shift(1)).abs(); lcp = (d["Low"]-d["Close"].shift(1)).abs()
    tr = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean(); d["ATR_PCT"] = (d["ATR_14"]/d["Close"])*100
    # Bollinger
    bb_mid = d["Close"].rolling(20).mean(); bb_std = d["Close"].rolling(20).std()
    d["BB_UPPER"] = bb_mid + 2*bb_std; d["BB_LOWER"] = bb_mid - 2*bb_std
    d["BB_WIDTH"] = ((d["BB_UPPER"]-d["BB_LOWER"])/bb_mid)*100
    d["BB_PCT_B"] = (d["Close"]-d["BB_LOWER"]) / (d["BB_UPPER"]-d["BB_LOWER"]).replace(0,np.nan)
    # Volume
    d["VOL_RATIO"] = d["Volume"] / d["Volume"].rolling(20).mean().replace(0,np.nan)
    # MACD
    e12 = d["Close"].ewm(span=12,adjust=False).mean(); e26 = d["Close"].ewm(span=26,adjust=False).mean()
    d["MACD"] = e12-e26; d["MACD_SIGN"] = d["MACD"].ewm(span=9,adjust=False).mean()
    d["MACD_HIST"] = d["MACD"]-d["MACD_SIGN"]
    # Stochastic
    l14 = d["Low"].rolling(14).min(); h14 = d["High"].rolling(14).max()
    d["STOCH_K"] = ((d["Close"]-l14)/(h14-l14).replace(0,np.nan))*100
    d["STOCH_D"] = d["STOCH_K"].rolling(3).mean()
    # OBV
    d["OBV"] = (np.sign(d["Close"].diff())*d["Volume"]).cumsum()
    # Retornos
    for p, name in [(1,"RETORNO_1D"),(5,"RETORNO_5D"),(20,"RETORNO_20D"),(60,"RETORNO_60D")]:
        d[name] = d["Close"].pct_change(p)*100
    # 52W
    d["MAX_52W"] = d["Close"].rolling(252).max(); d["MIN_52W"] = d["Close"].rolling(252).min()
    d["DIST_MAX"] = ((d["Close"]-d["MAX_52W"])/d["MAX_52W"])*100
    d["DIST_MIN"] = ((d["Close"]-d["MIN_52W"])/d["MIN_52W"])*100
    # Vol Hist
    d["VOLATILIDADE_20D"] = d["Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    return d

print("✅ Funções definidas.")

# COMMAND ----------

# DBTITLE 1,Download
print(f"🔽 Baixando: {DATA_INICIO.strftime('%d/%m/%Y')} → {DATA_CORTE.strftime('%d/%m/%Y')}\n")
DADOS_BRUTOS = baixar_dados(TICKERS_B3, DATA_INICIO, DATA_CORTE)
DADOS_IBOV   = DADOS_BRUTOS.pop("^BVSP", None)
DADOS_ACOES  = dict(DADOS_BRUTOS)
print(f"\n✅ {len(DADOS_ACOES)} ações com dados")

# COMMAND ----------

# DBTITLE 1,Indicadores
print("📐 Calculando indicadores...\n")
DADOS_COM_IND = {}
for tk, df in DADOS_ACOES.items():
    try: DADOS_COM_IND[tk] = calcular_indicadores(df)
    except Exception: pass
if DADOS_IBOV is not None: DADOS_IBOV = calcular_indicadores(DADOS_IBOV)
print(f"✅ {len(DADOS_COM_IND)} ativos com indicadores.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌡️ Regime de Mercado

# COMMAND ----------

# DBTITLE 1,Detecção de Regime
def detectar_regime(ibov_df):
    if ibov_df is None or len(ibov_df) < 50:
        return "NEUTRO", "Dados insuficientes"
    ult = ibov_df.iloc[-1]
    acima_mm200 = ult["Close"] > ult.get("MM200", ult["Close"])
    ret60 = ult.get("RETORNO_60D", 0); rsi = ult.get("RSI_14", 50); dist_max = ult.get("DIST_MAX", 0)
    pb = pe = 0
    if acima_mm200: pb += 1
    else: pe += 1
    if ret60 > 10: pb += 1
    elif ret60 < -10: pe += 1
    if rsi > 55: pb += 1
    elif rsi < 45: pe += 1
    if dist_max > -10: pb += 1
    elif dist_max < -25: pe += 1
    if pb >= 3: return "BULL", f"IBOV em alta (RSI={rsi:.1f}, Ret60d={ret60:.1f}%)"
    elif pe >= 3: return "BEAR", f"IBOV em queda/pânico (RSI={rsi:.1f}, Ret60d={ret60:.1f}%)"
    else: return "NEUTRO", f"IBOV lateral (RSI={rsi:.1f}, Ret60d={ret60:.1f}%)"

REGIME, REGIME_DESC = detectar_regime(DADOS_IBOV)
emap = {"BULL":"🟢","BEAR":"🔴","NEUTRO":"🟡"}
print(f"{emap.get(REGIME,'⚪')} REGIME: {REGIME} — {REGIME_DESC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Estratégias BNF

# COMMAND ----------

# DBTITLE 1,5 Estratégias de Kotegawa
def est1_reversao(df, ticker):
    if len(df) < 30: return None
    u = df.iloc[-1]; kairi = u.get("KAIRI_25",0); rsi = u.get("RSI_14",50)
    vol = u.get("VOL_RATIO",1); ret5 = u.get("RETORNO_5D",0)
    if not (kairi < -20 and rsi < 40): return None
    s = 0
    if kairi < -30: s += 40
    elif kairi < -25: s += 25
    else: s += 15
    if rsi < 25: s += 25
    elif rsi < 30: s += 15
    else: s += 10
    if vol > 1.2: s += 20
    if ret5 < -10: s += 15
    mm25 = u.get("MM25", u["Close"]*1.20); alvo = mm25
    pot = ((alvo - u["Close"])/u["Close"])*100
    return {"ticker":ticker,"estrategia":"1 — Reversão à Média (Kairi)","score":min(s,100),
            "preco":round(u["Close"],2),"kairi":round(kairi,2),"rsi":round(rsi,2),
            "vol_ratio":round(vol,2),"ret20d":round(u.get("RETORNO_20D",0),2),
            "alvo":round(alvo,2),"potencial":round(pot,2),"regime_alvo":["BEAR","NEUTRO","BULL"]}


def est2_vola_bear(df, ticker, regime):
    if len(df) < 30 or regime == "BULL": return None
    u = df.iloc[-1]; atr = u.get("ATR_PCT",0); bbL = u.get("BB_LOWER",0)
    bbW = u.get("BB_WIDTH",0); rsi = u.get("RSI_14",50); ret20 = u.get("RETORNO_20D",0)
    close = u["Close"]
    if not (atr > 4.0 and rsi < 38 and ret20 < -10): return None
    s = 0
    if atr > 8: s += 35
    elif atr > 6: s += 25
    else: s += 15
    if close <= bbL*1.02: s += 25
    if rsi < 25: s += 20
    elif rsi < 30: s += 12
    if ret20 < -25: s += 20
    elif ret20 < -15: s += 12
    if bbW > 30: s += 10
    mm25 = u.get("MM25", close*1.10); alvo = close + (mm25-close)*0.5
    pot = ((alvo-close)/close)*100
    return {"ticker":ticker,"estrategia":"2 — Volatilidade + Pânico (Bear)","score":min(s,100),
            "preco":round(close,2),"atr_pct":round(atr,2),"bb_width":round(bbW,2),
            "rsi":round(rsi,2),"ret20d":round(ret20,2),"alvo":round(alvo,2),
            "potencial":round(pot,2),"regime_alvo":["BEAR","NEUTRO"]}


def est3_atraso(df, ticker, ibov_df, regime):
    if len(df) < 65 or regime != "BULL": return None
    u = df.iloc[-1]; ret60a = u.get("RETORNO_60D",0); rsi = u.get("RSI_14",50)
    close = u["Close"]; mm55 = u.get("MM55",0); mm9 = u.get("MM9",0); mm21 = u.get("MM21",0)
    vr = u.get("VOL_RATIO",1)
    ret60i = ibov_df.iloc[-1].get("RETORNO_60D",0) if ibov_df is not None and len(ibov_df)>=60 else 0
    if ret60i < 8: return None
    lag = ret60a/ret60i if ret60i != 0 else 1
    if lag >= 0.5: return None
    if mm55 > 0 and close < mm55*0.95: return None
    if not (35 <= rsi <= 65): return None
    s = min(int((0.5-lag)*100),40)
    if ret60i > 20: s += 20
    elif ret60i > 15: s += 12
    if vr > 1.3: s += 15
    if mm9 > mm21: s += 15
    if rsi > 50: s += 10
    alvo = close*(1+(ret60i*0.75-ret60a)/100); pot = ((alvo-close)/close)*100
    return {"ticker":ticker,"estrategia":"3 — Atrasados no Rally (Bull)","score":min(s,100),
            "preco":round(close,2),"ret60_acao":round(ret60a,2),"ret60_ibov":round(ret60i,2),
            "lag_ratio":round(lag,2),"rsi":round(rsi,2),"vol_ratio":round(vr,2),
            "alvo":round(alvo,2),"potencial":round(pot,2),"regime_alvo":["BULL"]}


def est4_exaustao(df, ticker):
    if len(df) < 25: return None
    u = df.iloc[-1]; u5 = df.iloc[-5:]; close = u["Close"]
    ret20 = u.get("RETORNO_20D",0); rsi = u.get("RSI_14",50)
    v5 = u5["Volume"].mean() if "Volume" in u5.columns else 0
    v20 = df["Volume"].iloc[-20:].mean() if "Volume" in df.columns else 1
    vr = v5/v20 if v20>0 else 1
    if len(u5) < 5: return None
    ret5r = ((u5["Close"].iloc[-1]-u5["Close"].iloc[0])/u5["Close"].iloc[0])*100
    rng = u["High"]-u["Low"]; pf = ((close-u["Low"])/rng) if rng > 0 else 0.5
    if not (ret20 < -15 and vr < 0.60): return None
    s = 0
    if ret20 < -25: s += 30
    else: s += 20
    if vr < 0.40: s += 30
    elif vr < 0.50: s += 20
    else: s += 10
    if abs(ret5r) < 5: s += 20
    if pf > 0.6: s += 15
    if rsi < 35: s += 5
    mm25 = u.get("MM25", close*1.15); alvo = close+(mm25-close)*0.5
    pot = ((alvo-close)/close)*100
    return {"ticker":ticker,"estrategia":"4 — Exaustão de Volume","score":min(s,100),
            "preco":round(close,2),"vol_ratio":round(vr,2),"ret20d":round(ret20,2),
            "rsi":round(rsi,2),"alvo":round(alvo,2),"potencial":round(pot,2),
            "regime_alvo":["BEAR","NEUTRO","BULL"]}


def est5_sniper(df, ticker):
    if len(df) < 60: return None
    u = df.iloc[-1]; a3 = df.iloc[-4:-1]
    close = u["Close"]; mm9 = u.get("MM9",0); mm21 = u.get("MM21",0)
    mm55 = u.get("MM55",0); rsi = u.get("RSI_14",50); vr = u.get("VOL_RATIO",1); kairi = u.get("KAIRI_25",0)
    cruz = False
    if len(a3)>=2 and "MM9" in a3.columns and "MM21" in a3.columns:
        for i in range(len(a3)-1):
            m9b=a3["MM9"].iloc[i]; m21b=a3["MM21"].iloc[i]
            m9d=a3["MM9"].iloc[i+1]; m21d=a3["MM21"].iloc[i+1]
            if not (pd.isna(m9b) or pd.isna(m21b)):
                if m9b<=m21b and m9d>m21d: cruz=True; break
    mm9_up = mm9>mm21*1.001 if mm9>0 and mm21>0 else False
    ok_c = cruz or (mm9_up and rsi>48)
    if not (ok_c and 42<=rsi<=62 and (-15<=kairi<=5)): return None
    if mm55 > 0 and close < mm55*0.98: return None
    s = 35 if cruz else 15
    if 48<=rsi<=55: s+=25
    elif 42<=rsi<=62: s+=15
    if vr>=1.1: s+=20
    if mm55>0 and close>mm55*0.98: s+=10
    if -5<=kairi<=0: s+=10
    alvo = close*1.08; pot = 8.0
    return {"ticker":ticker,"estrategia":"5 — Sniper de Baixo Risco","score":min(s,100),
            "preco":round(close,2),"kairi":round(kairi,2),"rsi":round(rsi,2),
            "vol_ratio":round(vr,2),"alvo":round(alvo,2),"potencial":round(pot,2),
            "regime_alvo":["BULL","NEUTRO","BEAR"]}

print("✅ 5 estratégias BNF definidas.")

# COMMAND ----------

# DBTITLE 1,Varredura e Seleção do Campeão
print(f"🔍 Varrendo {len(DADOS_COM_IND)} ativos | Regime: {REGIME}\n")
resultados_todos = []
for ticker, df in DADOS_COM_IND.items():
    for fn in [
        lambda d,t: est1_reversao(d,t),
        lambda d,t: est2_vola_bear(d,t,REGIME),
        lambda d,t: est3_atraso(d,t,DADOS_IBOV,REGIME),
        lambda d,t: est4_exaustao(d,t),
        lambda d,t: est5_sniper(d,t),
    ]:
        try:
            r = fn(df, ticker)
            if r and REGIME in r.get("regime_alvo",[REGIME]): resultados_todos.append(r)
        except Exception: pass

resultados_todos.sort(key=lambda x: x["score"], reverse=True)

if not resultados_todos:
    print("⚠️ Nenhuma oportunidade BNF detectada.")
    MELHOR = None
else:
    MELHOR = resultados_todos[0]
    print(f"✅ {len(resultados_todos)} candidatos encontrados.")
    print(f"\n🏆 MELHOR: {MELHOR['ticker'].replace('.SA','')} | Score: {MELHOR['score']} | {MELHOR['estrategia']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 TOP 10 — Ranking dos Candidatos

# COMMAND ----------

# DBTITLE 1,Top 10 HTML
ECORES = {"1":"#FF6B6B","2":"#FF4444","3":"#00C851","4":"#FFBB33","5":"#33B5E5"}
EICONS = {"1":"📉","2":"🌩️","3":"🚀","4":"💤","5":"🎯"}
RCORES = {"BULL":"#00C851","BEAR":"#FF4444","NEUTRO":"#FFBB33"}

def sbadge(score):
    c = "#00C851" if score>=75 else ("#FFBB33" if score>=55 else "#FF6B6B")
    return f'<span style="background:{c};color:#000;padding:2px 10px;border-radius:12px;font-weight:800;">{score}</span>'

def html_top10(res, regime, dc, best_tk):
    rc = RCORES.get(regime,"#888")
    rlbl = {"BULL":"🟢 BULL","BEAR":"🔴 BEAR","NEUTRO":"🟡 NEUTRO"}
    if not res:
        return f'<div style="background:#1E1E2E;color:#CDD6F4;padding:24px;border-radius:16px;border:2px solid #FF4444;font-family:sans-serif;"><h2 style="color:#FF4444;">⚠️ Nenhum padrão BNF detectado em {dc.strftime("%d/%m/%Y")}</h2></div>'
    top = res[:10]
    h = f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;padding:24px;border-radius:16px;">
<h2 style="color:#CBA6F7;margin:0 0 6px;">🏅 TOP 10 Candidatos BNF</h2>
<p style="color:#888;font-size:13px;margin:0 0 16px;">Data: <strong>{dc.strftime("%d/%m/%Y")}</strong> | Regime: <strong style="color:{rc}">{rlbl.get(regime)}</strong> | {len(res)} candidatos</p>
<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="background:#313244;color:#CBA6F7;">
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:center;">#</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:left;">Ativo</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:left;">Estratégia</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">Score</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">Preço</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">Alvo</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">Potencial</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">RSI</th>
<th style="padding:10px 12px;border-bottom:2px solid #45475A;text-align:right;">Kairi</th>
</tr></thead><tbody>'''
    for i, r in enumerate(top):
        tk = r["ticker"].replace(".SA",""); best = r["ticker"]==best_tk
        bg = "#1A3A2A" if best else ("#2A2A3E" if i%2==0 else "#252535")
        bd = "3px solid #00C851" if best else "3px solid transparent"
        ec = ECORES.get(r["estrategia"][0],"#888")
        pot = r.get("potencial",0); pc = "#00C851" if pot>0 else "#FF4444"
        cr = "👑 " if best else f"{i+1}."
        kv = r.get("kairi",None)
        ks = f'{kv:.1f}%' if isinstance(kv,float) else "—"
        h += f'''<tr style="background:{bg};border-left:{bd};">
<td style="padding:9px 12px;text-align:center;color:#CBA6F7;font-weight:700;">{cr}</td>
<td style="padding:9px 12px;font-weight:700;color:{"#00C851" if best else "#CDD6F4"};">{tk}</td>
<td style="padding:9px 12px;color:{ec};font-size:12px;">{r["estrategia"]}</td>
<td style="padding:9px 12px;text-align:right;">{sbadge(r["score"])}</td>
<td style="padding:9px 12px;text-align:right;">R$ {r.get("preco",0):.2f}</td>
<td style="padding:9px 12px;text-align:right;color:#00C851;">R$ {r.get("alvo",0):.2f}</td>
<td style="padding:9px 12px;text-align:right;color:{pc};font-weight:700;">{pot:+.1f}%</td>
<td style="padding:9px 12px;text-align:right;">{r.get("rsi",0):.1f}</td>
<td style="padding:9px 12px;text-align:right;">{ks}</td>
</tr>'''
    h += '</tbody></table></div><p style="color:#666;font-size:11px;margin-top:10px;">👑 = melhor oportunidade selecionada</p></div>'
    return h

if resultados_todos:
    displayHTML(html_top10(resultados_todos, REGIME, DATA_CORTE, MELHOR["ticker"] if MELHOR else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏆 Laudo Completo — A Melhor Oportunidade

# COMMAND ----------

# DBTITLE 1,Coleta de Dados do Campeão
if MELHOR:
    tk_best = MELHOR["ticker"]
    df_best = DADOS_COM_IND.get(tk_best)
    ult     = df_best.iloc[-1] if df_best is not None else None

    def gv(k, d=0):
        return ult.get(k, d) if ult is not None else d

    preco      = MELHOR.get("preco", 0)
    alvo       = MELHOR.get("alvo", 0)
    stop_price = round(preco * 0.90, 2)
    potencial  = MELHOR.get("potencial", 0)
    estrategia = MELHOR.get("estrategia", "")
    score      = MELHOR.get("score", 0)

    rsi = gv("RSI_14", 50); kairi = gv("KAIRI_25", 0); atr_pct = gv("ATR_PCT", 0)
    atr_14 = gv("ATR_14", 0); bb_width = gv("BB_WIDTH", 0); bb_upper = gv("BB_UPPER", 0)
    bb_lower = gv("BB_LOWER", 0); bb_pct_b = gv("BB_PCT_B", 0.5); vol_ratio = gv("VOL_RATIO", 1)
    ret1d = gv("RETORNO_1D", 0); ret5d = gv("RETORNO_5D", 0); ret20d = gv("RETORNO_20D", 0)
    ret60d = gv("RETORNO_60D", 0); mm9 = gv("MM9", 0); mm21 = gv("MM21", 0); mm25 = gv("MM25", 0)
    mm55 = gv("MM55", 0); mm200 = gv("MM200", 0); dist_max = gv("DIST_MAX", 0)
    dist_min = gv("DIST_MIN", 0); macd_hist = gv("MACD_HIST", 0); macd_val = gv("MACD", 0)
    stoch_k = gv("STOCH_K", 50); stoch_d = gv("STOCH_D", 50); vol20d = gv("VOLATILIDADE_20D", 0)

    mm_alta  = (mm9>mm21>mm55) if all([mm9,mm21,mm55]) else False
    mm_baixa = (mm9<mm21<mm55) if all([mm9,mm21,mm55]) else False
    acima200 = preco > mm200 if mm200 > 0 else False

    INVESTIMENTO = 1000.0
    qtd_acoes  = int(INVESTIMENTO / preco) if preco > 0 else 0
    custo_real = round(qtd_acoes * preco, 2)
    alvo_total = round(qtd_acoes * alvo, 2)
    stop_total = round(qtd_acoes * stop_price, 2)
    lucro_alvo = round(alvo_total - custo_real, 2)
    perda_stop = round(stop_total - custo_real, 2)

    print(f"✅ {tk_best}: preço={preco}, alvo={alvo}, stop={stop_price}")
    print(f"   {qtd_acoes} ações | custo={custo_real} | lucro={lucro_alvo} | perda={perda_stop}")

# COMMAND ----------

# DBTITLE 1,Laudo HTML — 4 Pilares
def laudo_html():
    if not MELHOR: return "<p>Sem recomendação.</p>"
    tk = MELHOR["ticker"].replace(".SA","")
    rc = RCORES.get(REGIME,"#888")
    rlbl = {"BULL":"🟢 BULL","BEAR":"🔴 BEAR","NEUTRO":"🟡 NEUTRO"}
    sc_c = "#00C851" if score>=75 else ("#FFBB33" if score>=55 else "#FF6B6B")
    ec = ECORES.get(estrategia[0],"#888"); ei = EICONS.get(estrategia[0],"📊")
    rr = abs(potencial/10)

    # Rótulos e cores
    rsi_l = "Oversold Severo" if rsi<30 else ("Oversold" if rsi<40 else ("Neutro" if rsi<60 else "Overbought"))
    rsi_c = "#00C851" if rsi<40 else ("#FFBB33" if rsi<60 else "#FF4444")
    kai_l = "Extremamente Oversold" if kairi<-30 else ("Oversold Severo" if kairi<-20 else ("Oversold" if kairi<-10 else "Normal"))
    kai_c = "#00C851" if kairi<-20 else ("#FFBB33" if kairi<0 else "#FF4444")
    atr_l = "Volatilidade Extrema" if atr_pct>8 else ("Alta Volatilidade" if atr_pct>4 else "Vol. Moderada")
    atr_c = "#FF4444" if atr_pct>6 else ("#FFBB33" if atr_pct>3 else "#00C851")
    bb_l  = "Tocando Banda Inferior" if bb_pct_b<0.1 else ("Próx. Inferior" if bb_pct_b<0.25 else ("Neutro" if bb_pct_b<=0.75 else "Próx. Superior"))
    bb_c  = "#00C851" if bb_pct_b<0.25 else ("#FFBB33" if bb_pct_b<=0.75 else "#FF4444")
    mc_l  = "Histograma ↑ Positivo" if macd_hist>0 else "Histograma ↓ Negativo"
    mc_c  = "#00C851" if macd_hist>0 else "#FF4444"
    st_l  = "Sobrevenda <20" if stoch_k<20 else ("Sobrecompra >80" if stoch_k>80 else "Neutro")
    st_c  = "#00C851" if stoch_k<20 else ("#FF4444" if stoch_k>80 else "#FFBB33")
    mm_l  = "✅ Alinhadas ALTA" if mm_alta else ("⚠️ Alinhadas BAIXA" if mm_baixa else "⚡ Mistas")
    m200_l= "✅ Acima MM200" if acima200 else "⚠️ Abaixo MM200"

    # IBOV
    ibov_r60 = DADOS_IBOV.iloc[-1].get("RETORNO_60D",0) if DADOS_IBOV is not None and len(DADOS_IBOV)>=60 else 0
    ibov_rsi = DADOS_IBOV.iloc[-1].get("RSI_14",50) if DADOS_IBOV is not None else 50
    ibov_cl  = DADOS_IBOV.iloc[-1].get("Close",0) if DADOS_IBOV is not None else 0
    ibov_m200= DADOS_IBOV.iloc[-1].get("MM200",0) if DADOS_IBOV is not None else 0

    def sec(t, c, ic, body):
        return f'<div style="background:#252535;border-radius:10px;padding:14px;margin-bottom:14px;border-left:4px solid {c};"><h3 style="color:{c};margin:0 0 8px;font-size:15px;">{ic} {t}</h3>{body}</div>'

    p1 = f'''<p style="color:#888;font-size:12px;margin:0 0 6px;">O <strong>Kairi Ritsu</strong> é o indicador principal de BNF — mede o desvio percentual do preço em relação à MM25.</p>
<ul style="margin:0;padding-left:18px;line-height:1.9;font-size:13px;color:#BAC2DE;">
<li><strong style="color:#FF6B6B;">Kairi vs MM25:</strong> Preço {abs(kairi):.1f}% {"ABAIXO" if kairi<0 else "ACIMA"} da MM25 (Kairi={kairi:.1f}%). {"Zona de pânico — desvios acima de 20% têm alta taxa histórica de reversão dentro de 5-20 pregões." if kairi<-20 else "Monitorar convergência com MM25."}</li>
<li><strong style="color:#FF6B6B;">ATR% (14d):</strong> {atr_pct:.1f}% de amplitude média diária (R$ {atr_14:.2f}). {"Alta volatilidade = maior amplitude = maior lucro potencial em reversões." if atr_pct>4 else "Volatilidade moderada."}</li>
<li><strong style="color:#FF6B6B;">Bollinger Bands (20p, 2σ):</strong> Width={bb_width:.1f}% | %B={bb_pct_b:.2f} — {bb_l}. {"Preço na banda inferior — evento estatisticamente raro (~5% do tempo), frequentemente precede reversão ou estabilização." if bb_pct_b<0.25 else ""}</li>
<li><strong style="color:#FF6B6B;">Vol. Histórica Anualizada (20d):</strong> {vol20d:.1f}% a.a. {"Alta volatilidade — campo fértil para estratégias de reversão." if vol20d>40 else "Dentro do padrão histórico."}</li>
</ul>'''

    p2 = f'''<p style="color:#888;font-size:12px;margin:0 0 6px;">Indicadores de momentum validados academicamente: RSI (Wilder,1978), MACD (Appel,1979), Estocástico (Lane,1950s).</p>
<ul style="margin:0;padding-left:18px;line-height:1.9;font-size:13px;color:#BAC2DE;">
<li><strong style="color:#33B5E5;">RSI 14 (Wilder):</strong> {rsi:.1f} — <b style="color:{rsi_c};">{rsi_l}</b>. {"RSI abaixo de 40 indica sobrevenda prolongada — vendedores exageraram." if rsi<40 else "RSI neutro: ideal para entrada sem risco de sobrecompra." if 40<=rsi<=60 else "RSI elevado — atenção ao risco de realização."}</li>
<li><strong style="color:#33B5E5;">MACD (12,26,9):</strong> MACD={macd_val:.3f} | Hist={macd_hist:.3f} — <b style="color:{mc_c};">{mc_l}</b>. {"Histograma positivo: EMA12 superando EMA26 — mudança de momentum em curso." if macd_hist>0 else "Aguardar cruzamento MACD/sinal para confirmação."}</li>
<li><strong style="color:#33B5E5;">Estocástico (14,3):</strong> %K={stoch_k:.1f} | %D={stoch_d:.1f} — <b style="color:{st_c};">{st_l}</b>. {"%K<20 é zona clássica de sobrevenda — sinal de reversão histórico de George Lane." if stoch_k<20 else ""}</li>
<li><strong style="color:#33B5E5;">Estrutura MMs:</strong> {mm_l} | {m200_l}<br>MM9={mm9:.2f} | MM21={mm21:.2f} | MM25={mm25:.2f} | MM55={mm55:.2f} | MM200={mm200:.2f}</li>
</ul>'''

    p3 = f'''<ul style="margin:0;padding-left:18px;line-height:1.9;font-size:13px;color:#BAC2DE;">
<li><strong style="color:#FFBB33;">Volume Ratio:</strong> {vol_ratio:.2f}x a média de 20 dias. {"Volume ACIMA da média confirma participação real — pânico genuíno = oportunidade genuína." if vol_ratio>1.2 else "Volume ABAIXO da média — esgotamento dos vendedores. Sem vendedores, qualquer compra move o preço." if vol_ratio<0.6 else "Volume próximo à média — monitorar."}</li>
<li><strong style="color:#FFBB33;">Performance:</strong> 1d={ret1d:+.1f}% | 5d={ret5d:+.1f}% | 20d={ret20d:+.1f}% | 60d={ret60d:+.1f}% | Topo 52W: {dist_max:.1f}% | Fundo 52W: {dist_min:+.1f}%</li>
</ul>'''

    p4 = f'''<ul style="margin:0;padding-left:18px;line-height:1.9;font-size:13px;color:#BAC2DE;">
<li><strong style="color:{rc};">Regime IBOV — {rlbl.get(REGIME)}:</strong> {REGIME_DESC}<br>IBOV: {ibov_cl:.0f} pts | RSI={ibov_rsi:.1f} | Ret60d={ibov_r60:+.1f}% | {"Acima" if ibov_cl>ibov_m200 else "Abaixo"} da MM200 ({ibov_m200:.0f})<br>Estratégia <em>{estrategia}</em> é adequada ao regime {REGIME}.</li>
</ul>'''

    painel = f'''<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;font-size:12px;">
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">RSI 14:</span> <strong style="color:{rsi_c};">{rsi:.1f} — {rsi_l}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Kairi 25:</span> <strong style="color:{kai_c};">{kairi:.1f}% — {kai_l}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">ATR%:</span> <strong style="color:{atr_c};">{atr_pct:.1f}% — {atr_l}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">BB Width/%B:</span> <strong>{bb_width:.1f}% / {bb_pct_b:.2f}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">MACD Hist:</span> <strong style="color:{mc_c};">{macd_hist:.4f}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Stoch %K/%D:</span> <strong style="color:{st_c};">{stoch_k:.1f}/{stoch_d:.1f}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Vol Ratio:</span> <strong>{vol_ratio:.2f}x</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Ret 1d/5d/20d:</span> <strong>{ret1d:+.1f}%/{ret5d:+.1f}%/{ret20d:+.1f}%</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Vol Hist 20d:</span> <strong>{vol20d:.1f}% a.a.</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">Topo/Fundo 52W:</span> <strong>{dist_max:.1f}% / {dist_min:+.1f}%</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">MMs:</span> <strong>{mm_l}</strong></div>
<div style="background:#1E1E2E;border-radius:7px;padding:8px;"><span style="color:#888;">MM200:</span> <strong>{"✅ Acima" if acima200 else "⚠️ Abaixo"}</strong></div>
</div>'''

    return f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;padding:28px;border-radius:16px;">
<div style="border:2px solid #CBA6F7;border-radius:14px;padding:20px;margin-bottom:24px;background:linear-gradient(135deg,#2A1F4E,#1E1E2E);">
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
<div>
<div style="font-size:11px;color:#888;letter-spacing:2px;margin-bottom:4px;">🏆 MELHOR OPORTUNIDADE BNF — {DATA_CORTE.strftime("%d/%m/%Y")}</div>
<div style="font-size:42px;font-weight:900;color:#CBA6F7;">{tk}</div>
<div style="font-size:14px;color:{ec};margin-top:4px;">{ei} {estrategia}</div>
</div>
<div style="text-align:right;">
<div style="font-size:11px;color:#888;margin-bottom:4px;">Score BNF</div>
<div style="font-size:52px;font-weight:900;color:{sc_c};line-height:1;">{score}</div>
<div style="font-size:11px;color:#888;">/100</div>
</div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:18px;">
<div style="background:#1E1E2E;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:10px;color:#888;">💰 ENTRADA</div><div style="font-size:18px;font-weight:700;">R$ {preco:.2f}</div></div>
<div style="background:#1E1E2E;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:10px;color:#888;">🎯 ALVO</div><div style="font-size:18px;font-weight:700;color:#00C851;">R$ {alvo:.2f}</div><div style="font-size:11px;color:#00C851;">{potencial:+.1f}%</div></div>
<div style="background:#1E1E2E;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:10px;color:#888;">🛑 STOP (-10%)</div><div style="font-size:18px;font-weight:700;color:#FF4444;">R$ {stop_price:.2f}</div></div>
<div style="background:#1E1E2E;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:10px;color:#888;">⚖️ R/R</div><div style="font-size:18px;font-weight:700;color:#CBA6F7;">{rr:.1f}:1</div></div>
<div style="background:#1E1E2E;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:10px;color:#888;">🌡️ REGIME</div><div style="font-size:18px;font-weight:700;color:{rc};">{rlbl.get(REGIME)}</div></div>
</div>
</div>
{sec("Pilar 1 — Volatilidade BNF (Kairi, ATR, Bollinger)","#FF6B6B","🌊",p1)}
{sec("Pilar 2 — Momentum & Tendência (RSI, MACD, Estocástico, MMs)","#33B5E5","📐",p2)}
{sec("Pilar 3 — Volume e Performance","#FFBB33","📦",p3)}
{sec("Pilar 4 — Contexto de Mercado (IBOV)",rc,"🏢",p4)}
{sec("📊 Painel Consolidado","#89B4FA","📊",painel)}
<div style="background:#313244;border-radius:8px;padding:12px;font-size:11px;color:#666;">⚠️ Educacional. Não é recomendação de investimento. Use stop-loss.</div>
</div>'''

if MELHOR:
    displayHTML(laudo_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💰 Simulação R$ 1.000

# COMMAND ----------

# DBTITLE 1,Simulação HTML
def sim_html():
    if not MELHOR: return "<p>Sem recomendação.</p>"
    tk = MELHOR["ticker"].replace(".SA",""); rr = abs(potencial/10)
    return f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;padding:24px;border-radius:16px;">
<h2 style="color:#A6E3A1;margin:0 0 18px;">💰 Simulação R$ {INVESTIMENTO:,.0f} em {tk}</h2>
<p style="color:#888;font-size:13px;margin:0 0 18px;">Compra em {DATA_CORTE.strftime("%d/%m/%Y")} | Preço: R$ {preco:.2f}</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px;">
<div style="background:#1A2F1F;border:1px solid #00C851;border-radius:10px;padding:16px;">
<div style="font-size:10px;color:#888;">Ações Compradas</div>
<div style="font-size:32px;font-weight:900;color:#00C851;">{qtd_acoes}</div>
<div style="font-size:11px;color:#888;">com R$ {INVESTIMENTO:,.0f}</div>
</div>
<div style="background:#1A2F1F;border:1px solid #00C851;border-radius:10px;padding:16px;">
<div style="font-size:10px;color:#888;">Custo Real</div>
<div style="font-size:32px;font-weight:900;">R$ {custo_real:.2f}</div>
<div style="font-size:11px;color:#888;">{qtd_acoes} × R$ {preco:.2f}</div>
</div>
<div style="background:#122A1A;border:1px solid #00C851;border-radius:10px;padding:16px;">
<div style="font-size:10px;color:#888;">🎯 Se Alvo Atingido</div>
<div style="font-size:32px;font-weight:900;color:#00C851;">R$ {alvo_total:.2f}</div>
<div style="font-size:11px;color:#00C851;">+R$ {lucro_alvo:.2f} ({potencial:+.1f}%)</div>
</div>
<div style="background:#2A1212;border:1px solid #FF4444;border-radius:10px;padding:16px;">
<div style="font-size:10px;color:#888;">🛑 Se Stop (-10%)</div>
<div style="font-size:32px;font-weight:900;color:#FF4444;">R$ {stop_total:.2f}</div>
<div style="font-size:11px;color:#FF4444;">R$ {perda_stop:.2f} (-10%)</div>
</div>
</div>
<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="background:#313244;color:#CBA6F7;">
<th style="padding:10px 14px;text-align:left;border-bottom:2px solid #45475A;">Cenário</th>
<th style="padding:10px 14px;text-align:right;border-bottom:2px solid #45475A;">Preço/ação</th>
<th style="padding:10px 14px;text-align:right;border-bottom:2px solid #45475A;">Total</th>
<th style="padding:10px 14px;text-align:right;border-bottom:2px solid #45475A;">Resultado</th>
<th style="padding:10px 14px;text-align:right;border-bottom:2px solid #45475A;">Retorno</th>
</tr></thead><tbody>
<tr style="background:#2A2A3E;">
<td style="padding:9px 14px;font-weight:600;">📥 Compra</td>
<td style="padding:9px 14px;text-align:right;">R$ {preco:.2f}</td>
<td style="padding:9px 14px;text-align:right;">R$ {custo_real:.2f}</td>
<td style="padding:9px 14px;text-align:right;color:#888;">—</td>
<td style="padding:9px 14px;text-align:right;color:#888;">0,00%</td>
</tr>
<tr style="background:#122A1A;border-left:3px solid #00C851;">
<td style="padding:9px 14px;font-weight:600;">✅ Alvo BNF</td>
<td style="padding:9px 14px;text-align:right;color:#00C851;">R$ {alvo:.2f}</td>
<td style="padding:9px 14px;text-align:right;color:#00C851;">R$ {alvo_total:.2f}</td>
<td style="padding:9px 14px;text-align:right;font-weight:700;color:#00C851;">R$ {lucro_alvo:+.2f}</td>
<td style="padding:9px 14px;text-align:right;font-weight:700;color:#00C851;">{potencial:+.1f}%</td>
</tr>
<tr style="background:#2A1212;border-left:3px solid #FF4444;">
<td style="padding:9px 14px;font-weight:600;">🛑 Stop -10%</td>
<td style="padding:9px 14px;text-align:right;color:#FF4444;">R$ {stop_price:.2f}</td>
<td style="padding:9px 14px;text-align:right;color:#FF4444;">R$ {stop_total:.2f}</td>
<td style="padding:9px 14px;text-align:right;font-weight:700;color:#FF4444;">R$ {perda_stop:+.2f}</td>
<td style="padding:9px 14px;text-align:right;font-weight:700;color:#FF4444;">-10,0%</td>
</tr>
</tbody></table></div>
<div style="background:#313244;border-radius:8px;padding:12px;margin-top:14px;font-size:12px;">
⚖️ R/R: para cada R$1 arriscado, potencial de R${rr:.1f} ({potencial:+.1f}%). {"✅ R/R favorável (≥2:1) — BNF operava assim." if rr>=2 else "⚠️ R/R abaixo de 2:1 — reduzir posição."}
</div></div>'''

if MELHOR:
    displayHTML(sim_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔬 Backtesting

# COMMAND ----------

# DBTITLE 1,Download Pós-Corte
HOJE = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
DIAS_APOS = (HOJE - DATA_CORTE).days
print(f"📅 Corte: {DATA_CORTE.strftime('%d/%m/%Y')} | Hoje: {HOJE.strftime('%d/%m/%Y')} | {DIAS_APOS} dias")

FAZER_BT = DIAS_APOS >= 3 and MELHOR is not None
AV = None

if FAZER_BT:
    print(f"✅ {DIAS_APOS} dias → executando backtest!")
    tk_best  = MELHOR["ticker"]
    DATA_POS = DATA_CORTE + timedelta(days=1)
    DADOS_POS    = baixar_dados([tk_best], DATA_POS, HOJE+timedelta(days=1), batch_size=5)
    DADOS_FULL_BT= baixar_dados([tk_best], DATA_INICIO, HOJE+timedelta(days=1), batch_size=5)
    print(f"✅ {len(DADOS_POS.get(tk_best,[]))} pregões pós-corte")
else:
    DADOS_POS={}; DADOS_FULL_BT={}
    print("🔵 Data recente — backtest ignorado.")

# COMMAND ----------

# DBTITLE 1,Avaliação
def avaliar(ticker, pe, alvo_p, dados_p):
    if ticker not in dados_p or len(dados_p[ticker])==0:
        return {"resultado":"❓ Sem dados","data_resultado":None,"retorno_realizado":None,
                "ultimo_preco":None,"max_preco":None,"min_preco":None,"dias_ate_resultado":None,"cor":"#888"}
    df = dados_p[ticker].sort_index(); sp = pe*0.90
    res="⏳ Ainda aberto"; dr=None; dias=None; cor="#FFBB33"
    for i, (dt, row) in enumerate(df.iterrows()):
        h=row.get("High",row["Close"]); l=row.get("Low",row["Close"])
        if l<=sp: res=f"🛑 Stop (≤R${sp:.2f})"; dr=dt; dias=i+1; cor="#FF4444"; break
        if h>=alvo_p: res=f"✅ Alvo (≥R${alvo_p:.2f})"; dr=dt; dias=i+1; cor="#00C851"; break
    up = df.loc[dr]["Close"] if dr is not None else df.iloc[-1]["Close"]
    ret = ((up-pe)/pe)*100
    return {"resultado":res,"data_resultado":dr.strftime("%d/%m/%Y") if dr else "—",
            "retorno_realizado":round(ret,2),"ultimo_preco":round(up,2),
            "max_preco":round(df["High"].max(),2) if "High" in df.columns else None,
            "min_preco":round(df["Low"].min(),2) if "Low" in df.columns else None,
            "dias_ate_resultado":dias,"cor":cor}

if FAZER_BT and MELHOR and tk_best in DADOS_POS:
    AV = avaliar(tk_best, preco, alvo, DADOS_POS)
    print(f"\n📊 RESULTADO: {AV['resultado']}")
    print(f"   Data: {AV['data_resultado']} | Retorno: {AV['retorno_realizado']:+.2f}%")
    print(f"   Máx: R${AV['max_preco']} | Mín: R${AV['min_preco']} | Dias: {AV['dias_ate_resultado']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Gráfico — Preço + RSI

# COMMAND ----------

# DBTITLE 1,Gráfico HTML
def graf_html(ticker, dados_f, dc, pe, alvo_p, stop_p, av):
    if ticker not in dados_f or len(dados_f[ticker])<5:
        return "<p style='color:#888;'>Dados insuficientes para gráfico.</p>"
    df = dados_f[ticker].sort_index()
    da = df[df.index<=pd.Timestamp(dc)]; dd = df[df.index>pd.Timestamp(dc)]
    tk = ticker.replace(".SA","")
    res_s = av.get("resultado","—") if av else "—"
    ret_r = av.get("retorno_realizado",None) if av else None
    ret_s = f"{ret_r:+.1f}%" if ret_r is not None else "Sem dados"
    cor_r = av.get("cor","#FFBB33") if av else "#CBA6F7"

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),gridspec_kw={"height_ratios":[3,1]},sharex=True)
    fig.patch.set_facecolor("#1E1E2E")
    for ax in [ax1,ax2]: ax.set_facecolor("#1E1E2E")

    if len(da)>0: ax1.plot(da.index, da["Close"], color="#89B4FA", linewidth=1.5, label="Pré-corte", zorder=3)
    if len(dd)>0:
        ax1.plot(dd.index, dd["Close"], color=cor_r, linewidth=2.5, label="Pós-corte", zorder=3)
        ax1.axvspan(pd.Timestamp(dc), df.index[-1], alpha=0.07, color="#CBA6F7", zorder=1)

    for p,c,l in [(25,"#F38BA8","MM25"),(55,"#A6E3A1","MM55"),(200,"#F9E2AF","MM200")]:
        if len(df)>=p: ax1.plot(df.index, df["Close"].rolling(p).mean(), color=c, linewidth=0.9, linestyle="--", alpha=0.7, label=l, zorder=2)

    ax1.axvline(x=pd.Timestamp(dc), color="#CBA6F7", linewidth=2, linestyle="--", label=f"Corte {dc.strftime('%d/%m/%Y')}", zorder=4)
    ax1.scatter([pd.Timestamp(dc)],[pe], color="#FAB387", s=120, zorder=7, label=f"Entrada R${pe:.2f}")
    ax1.axhline(y=alvo_p, color="#00C851", linewidth=1.5, linestyle="-.", label=f"Alvo R${alvo_p:.2f}", zorder=4)
    ax1.axhline(y=stop_p, color="#FF4444", linewidth=1.5, linestyle=":", label=f"Stop R${stop_p:.2f}", zorder=4)

    if av:
        drs = av.get("data_resultado","—")
        if drs and drs != "—":
            try:
                drd = pd.Timestamp(datetime.strptime(drs,"%d/%m/%Y"))
                idx = df.index.searchsorted(drd)
                if idx<len(df):
                    pr = df["Close"].iloc[idx]
                    mk = "^" if "✅" in res_s else ("v" if "🛑" in res_s else "o")
                    ax1.scatter([drd],[pr], color=cor_r, s=200, marker=mk, zorder=8, label=f"Resultado {drs}")
            except Exception: pass

    ax1.set_title(f"{tk}  |  {res_s}  |  Retorno: {ret_s}", color="#CDD6F4", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Preço (R$)", color="#CDD6F4", fontsize=10)
    ax1.grid(axis="y", color="#313244", linewidth=0.5, alpha=0.7)
    ax1.grid(axis="x", color="#313244", linewidth=0.3, alpha=0.4)
    ax1.spines[:].set_color("#45475A"); ax1.tick_params(colors="#CDD6F4")
    ax1.legend(loc="upper left", framealpha=0.3, facecolor="#313244", edgecolor="#45475A", labelcolor="#CDD6F4", fontsize=8)

    if "RSI_14" in df.columns:
        ax2.plot(df.index, df["RSI_14"], color="#CBA6F7", linewidth=1.2, label="RSI 14")
        ax2.axhline(y=70, color="#FF4444", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.axhline(y=30, color="#00C851", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.axhline(y=50, color="#45475A", linewidth=0.6, linestyle=":", alpha=0.5)
        ax2.fill_between(df.index, df["RSI_14"], 30, where=df["RSI_14"]<30, alpha=0.2, color="#00C851")
        ax2.fill_between(df.index, df["RSI_14"], 70, where=df["RSI_14"]>70, alpha=0.2, color="#FF4444")
        ax2.axvline(x=pd.Timestamp(dc), color="#CBA6F7", linewidth=1.5, linestyle="--", alpha=0.5)
        ax2.set_ylim(0,100); ax2.set_ylabel("RSI 14", color="#CDD6F4", fontsize=9)
        ax2.grid(axis="y", color="#313244", linewidth=0.5, alpha=0.5)
        ax2.spines[:].set_color("#45475A"); ax2.tick_params(colors="#CDD6F4")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.set_xlabel("Data", color="#CDD6F4", fontsize=10)
    plt.xticks(rotation=35, color="#CDD6F4", fontsize=9)
    plt.tight_layout(pad=1.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    cc = av.get("cor","#45475A") if av else "#CBA6F7"

    return f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;padding:24px;border-radius:16px;margin-top:8px;">
<h2 style="color:#CBA6F7;margin:0 0 16px;">📊 Gráfico — {tk}</h2>
<div style="background:#313244;border-radius:14px;padding:16px;border-left:4px solid {cc};">
<img src="data:image/png;base64,{img}" style="width:100%;border-radius:8px;display:block;"/>
</div>
<p style="color:#666;font-size:11px;margin-top:8px;">🟣=data de corte | 🟢=alvo | 🔴=stop | 🟠=entrada</p>
</div>'''

if FAZER_BT and DADOS_FULL_BT and MELHOR:
    displayHTML(graf_html(tk_best, DADOS_FULL_BT, DATA_CORTE, preco, alvo, stop_price, AV))
elif MELHOR and not FAZER_BT and DADOS_COM_IND.get(MELHOR["ticker"]) is not None:
    displayHTML(graf_html(MELHOR["ticker"], {MELHOR["ticker"]:DADOS_COM_IND[MELHOR["ticker"]]}, DATA_CORTE, preco, alvo, stop_price, None))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏁 Veredicto — O que teria acontecido?

# COMMAND ----------

# DBTITLE 1,Veredicto Final
def veredicto_html():
    if not MELHOR: return "<p>Sem recomendação.</p>"
    tk = MELHOR["ticker"].replace(".SA","")

    if AV is None:
        return f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;padding:24px;border-radius:16px;border:2px solid #FFBB33;">
<h2 style="color:#FFBB33;">🔵 Backtesting não disponível</h2>
<p>Data de corte {DATA_CORTE.strftime("%d/%m/%Y")} muito recente. Precisa de pelo menos 3 pregões após o corte.</p>
<hr style="border-color:#45475A;">
<p><strong>O que você faria:</strong> Comprar <strong>{qtd_acoes} ações</strong> de <strong>{tk}</strong> a R${preco:.2f} (custo: R${custo_real:.2f}), com alvo R${alvo:.2f} ({potencial:+.1f}%) e stop R${stop_price:.2f} (-10%).</p>
</div>'''

    res = AV.get("resultado","❓"); ret = AV.get("retorno_realizado",0) or 0
    dias = AV.get("dias_ate_resultado",None); dr = AV.get("data_resultado","—")
    maxp = AV.get("max_preco",0) or 0; minp = AV.get("min_preco",0) or 0
    cor = AV.get("cor","#888")
    vf = custo_real*(1+ret/100); lp = vf-custo_real
    is_win="✅" in res; is_stop="🛑" in res
    emoji = "🎉" if is_win else ("😰" if is_stop else "⏳")
    titulo = "O Alvo foi Atingido!" if is_win else ("Stop Acionado." if is_stop else "Operação em Andamento")
    mgp = ((maxp-preco)/preco*100) if preco>0 else 0
    mpp = ((minp-preco)/preco*100) if preco>0 else 0

    if is_win:
        narr = f"A compra de <strong>{qtd_acoes} ações</strong> de <strong>{tk}</strong> a R${preco:.2f} em {DATA_CORTE.strftime('%d/%m/%Y')} se mostrou <strong style='color:#00C851;'>acertada</strong>. Em <strong>{dias} pregões</strong> ({dr}), o preço atingiu o alvo de R${alvo:.2f}. Os R${custo_real:.2f} viraram R${vf:.2f}, gerando <strong>R${lp:+.2f} ({ret:+.1f}%)</strong>. A estratégia {estrategia} (Score {score}) entregou o resultado. Máxima no período: R${maxp:.2f} ({mgp:+.1f}%)."
        licao = "<li>✅ Kairi e indicadores funcionaram: reversão à média confirmada.</li><li>✅ Stop não acionado — operação limpa.</li><li>✅ Paciência + disciplina = resultado.</li>"
    elif is_stop:
        narr = f"A compra de <strong>{qtd_acoes} ações</strong> de <strong>{tk}</strong> a R${preco:.2f} em {DATA_CORTE.strftime('%d/%m/%Y')} acionou o stop. Em <strong>{dias} pregões</strong> ({dr}), o preço caiu abaixo de R${stop_price:.2f}. Os R${custo_real:.2f} ficaram em R${vf:.2f} — perda de <strong style='color:#FF4444;'>R${lp:+.2f} ({ret:+.1f}%)</strong>. O stop de -10% preservou 90% do capital. Mínima: R${minp:.2f} ({mpp:+.1f}%)."
        licao = "<li>🛑 O mercado venceu desta vez — setup não confirmado.</li><li>✅ Stop funcionou: capital protegido para próxima oportunidade.</li><li>💡 BNF: 'Stop é uma vitória sobre o ego. Preservar capital é regra nº1.'</li>"
    else:
        narr = f"Operação de <strong>{qtd_acoes} ações</strong> de <strong>{tk}</strong> a R${preco:.2f} em {DATA_CORTE.strftime('%d/%m/%Y')} <strong>ainda em aberto</strong>. Variação atual: <strong>{ret:+.1f}%</strong>. Máx: R${maxp:.2f} ({mgp:+.1f}%) | Mín: R${minp:.2f} ({mpp:+.1f}%). Alvo: R${alvo:.2f} | Stop: R${stop_price:.2f}."
        licao = "<li>⏳ Aguardar resolução com disciplina.</li><li>⚡ Não mova o stop por ansiedade — siga o plano.</li>"

    return f'''<div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;padding:28px;border-radius:16px;border:2px solid {cor};">
<div style="text-align:center;margin-bottom:24px;">
<div style="font-size:60px;">{emoji}</div>
<h1 style="color:{cor};margin:8px 0 4px;font-size:28px;">{titulo}</h1>
<p style="color:#888;font-size:14px;margin:0;">{tk} | Score BNF {score} | {estrategia}</p>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px;">
<div style="background:#313244;border-radius:10px;padding:14px;text-align:center;"><div style="font-size:10px;color:#888;">Investido</div><div style="font-size:22px;font-weight:700;">R$ {custo_real:.2f}</div></div>
<div style="background:#313244;border-radius:10px;padding:14px;text-align:center;border:1px solid {cor};"><div style="font-size:10px;color:#888;">Valor Final</div><div style="font-size:22px;font-weight:700;color:{cor};">R$ {vf:.2f}</div></div>
<div style="background:#313244;border-radius:10px;padding:14px;text-align:center;"><div style="font-size:10px;color:#888;">Resultado</div><div style="font-size:22px;font-weight:700;color:{cor};">R$ {lp:+.2f}</div><div style="font-size:13px;color:{cor};">{ret:+.1f}%</div></div>
<div style="background:#313244;border-radius:10px;padding:14px;text-align:center;"><div style="font-size:10px;color:#888;">Dias</div><div style="font-size:22px;font-weight:700;">{dias if dias else "—"}</div></div>
<div style="background:#1A2F1F;border-radius:10px;padding:14px;text-align:center;border:1px solid #00C851;"><div style="font-size:10px;color:#888;">Máxima</div><div style="font-size:22px;font-weight:700;color:#00C851;">R$ {maxp:.2f}</div><div style="font-size:12px;color:#00C851;">{mgp:+.1f}%</div></div>
<div style="background:#2A1212;border-radius:10px;padding:14px;text-align:center;border:1px solid #FF4444;"><div style="font-size:10px;color:#888;">Mínima</div><div style="font-size:22px;font-weight:700;color:#FF4444;">R$ {minp:.2f}</div><div style="font-size:12px;color:#FF4444;">{mpp:+.1f}%</div></div>
</div>
<div style="background:#252535;border-radius:10px;padding:16px;margin-bottom:16px;border-left:4px solid {cor};">
<h3 style="color:{cor};margin:0 0 10px;font-size:15px;">📖 O que aconteceu</h3>
<p style="margin:0;font-size:13px;line-height:1.8;color:#BAC2DE;">{narr}</p>
</div>
<div style="background:#252535;border-radius:10px;padding:16px;border-left:4px solid #CBA6F7;">
<h3 style="color:#CBA6F7;margin:0 0 10px;font-size:15px;">🧠 O que BNF diria</h3>
<ul style="margin:0;padding-left:18px;line-height:1.8;font-size:13px;color:#BAC2DE;">{licao}</ul>
</div>
</div>'''

if MELHOR:
    displayHTML(veredicto_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Tabela Pandas — Resumo

# COMMAND ----------

# DBTITLE 1,DataFrame de Resumo
if MELHOR:
    dr = {
        "ticker":[MELHOR["ticker"].replace(".SA","")],"estrategia":[estrategia],"score_bnf":[score],
        "data_corte":[DATA_CORTE.strftime("%d/%m/%Y")],"regime":[REGIME],
        "preco":[preco],"alvo":[alvo],"stop":[stop_price],"potencial_pct":[round(potencial,2)],
        "rsi_14":[round(rsi,2)],"kairi_25":[round(kairi,2)],"atr_pct":[round(atr_pct,2)],
        "bb_width":[round(bb_width,2)],"bb_pct_b":[round(bb_pct_b,3)],"vol_ratio":[round(vol_ratio,2)],
        "macd_hist":[round(macd_hist,4)],"stoch_k":[round(stoch_k,1)],"stoch_d":[round(stoch_d,1)],
        "ret_1d":[round(ret1d,2)],"ret_5d":[round(ret5d,2)],"ret_20d":[round(ret20d,2)],"ret_60d":[round(ret60d,2)],
        "vol_hist_20d_anual":[round(vol20d,1)],
        "mm9":[round(mm9,2)],"mm21":[round(mm21,2)],"mm25":[round(mm25,2)],"mm55":[round(mm55,2)],"mm200":[round(mm200,2)],
        "dist_topo_52w":[round(dist_max,2)],"dist_fundo_52w":[round(dist_min,2)],
        "qtd_acoes":[qtd_acoes],"custo_real":[custo_real],"alvo_total":[alvo_total],
        "stop_total":[stop_total],"lucro_alvo":[lucro_alvo],"perda_stop":[perda_stop],
    }
    if AV:
        dr["resultado_bt"]=[AV.get("resultado","N/A")]; dr["data_resultado"]=[AV.get("data_resultado","—")]
        dr["dias"]=[AV.get("dias_ate_resultado",None)]; dr["retorno_real_pct"]=[AV.get("retorno_realizado",None)]
        dr["max_pos"]=[AV.get("max_preco",None)]; dr["min_pos"]=[AV.get("min_preco",None)]
        ret_r = AV.get("retorno_realizado",0) or 0
        dr["valor_final"]=[round(custo_real*(1+ret_r/100),2)]; dr["lucro_perda"]=[round(custo_real*ret_r/100,2)]

    df_res = pd.DataFrame(dr)
    print(f"📋 Resumo: {MELHOR['ticker'].replace('.SA','')} | {DATA_CORTE.strftime('%d/%m/%Y')}\n")
    display(df_res.T.rename(columns={0:"Valor"}))
    try:
        path = f"/tmp/bnf_melhor_{DATA_CORTE.strftime('%Y%m%d')}.csv"
        df_res.to_csv(path, index=False); print(f"\n💾 CSV: {path}")
    except Exception as e: print(f"⚠️ {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏁 Fim — BNF Melhor Ação
# MAGIC
# MAGIC ### Resumo:
# MAGIC 1. ✅ ~130 ações B3 varridas com 5 estratégias BNF
# MAGIC 2. ✅ Indicadores: Kairi, RSI, ATR, Bollinger %B/Width, MACD, Estocástico, OBV, Vol. Histórica
# MAGIC 3. ✅ Regime IBOV detectado automaticamente
# MAGIC 4. ✅ **Ativo com maior score composto selecionado como "Melhor Compra"**
# MAGIC 5. ✅ Laudo em 4 pilares com justificativas completas
# MAGIC 6. ✅ Simulação R$1.000 com cenários alvo/stop
# MAGIC 7. ✅ Backtesting + gráfico preço/RSI + veredicto narrativo

# COMMAND ----------

# MAGIC %md
# MAGIC > ⚠️ **7 Princípios de BNF:** 1) Stop sempre | 2) Liquidez obrigatória | 3) O mercado está certo
# MAGIC > 4) Paciência | 5) Controle emocional | 6) Diversifique | 7) O preço reverte à média
