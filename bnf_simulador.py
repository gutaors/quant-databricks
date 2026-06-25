# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 🎯 BNF Simulator — Estratégias de Takashi Kotegawa para Ações Brasileiras
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 👤 Quem é Takashi Kotegawa (BNF)?
# MAGIC
# MAGIC **Takashi Kotegawa**, conhecido online como **BNF** (*Buy N' Forget*), é um dos traders mais lendários do Japão.
# MAGIC
# MAGIC Ele transformou aproximadamente **US$ 13.600** em mais de **US$ 150 milhões** em cerca de 8 anos,
# MAGIC operando exclusivamente na Bolsa de Valores do Japão (TSE), sem gestoras, sem cursos vendidos, sem sócios.
# MAGIC
# MAGIC ### 🧠 Filosofia Central
# MAGIC
# MAGIC > *"Não importa o que a empresa faz. Importa o que o preço faz."*
# MAGIC
# MAGIC BNF ignorava completamente fundamentos (balanços, notícias de CEOs, relatórios trimestrais).
# MAGIC Sua vantagem era **ler o pânico e a ganância** diretamente no preço e no volume.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Estratégias Implementadas
# MAGIC
# MAGIC | # | Nome | Mercado | Descrição |
# MAGIC |---|------|---------|-----------|
# MAGIC | 1 | **Reversão à Média (Kairi)** | Bear/Neutro | Ação despencou muito longe da MM25 → rebote esperado |
# MAGIC | 2 | **Volatilidade + Pânico (Bear)** | Bear | Alta volatilidade + oversold extremo = oportunidade de scalp |
# MAGIC | 3 | **Atrasados no Rally (Bull)** | Bull | Ações que ainda não subiram com o IBOV → vão subir em atraso |
# MAGIC | 4 | **Exaustão de Volume** | Qualquer | Volume seco após queda = vendedores esgotados → reversão |
# MAGIC | 5 | **Sniper de Baixo Risco** | Qualquer | Setup técnico perfeito: MM alinhadas + RSI neutro + candle de reversão |
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Instalação de Dependências

# COMMAND ----------

# MAGIC %pip install yfinance requests tqdm --quiet
# MAGIC

# COMMAND ----------

# DBTITLE 1,Instalação de Pacotes
# MAGIC %pip install yfinance requests tqdm --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Imports e Configuração Global
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from IPython.display import display, HTML
import time

warnings.filterwarnings("ignore")

print("✅ Imports realizados com sucesso!")
print(f"📅 Data e hora atual: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📅 INSIRA A DATA DE CORTE
# MAGIC
# MAGIC > **Instrução:** Informe a data de corte no widget abaixo (formato `YYYY-MM-DD`).
# MAGIC > O notebook vai buscar dados históricos até essa data e identificar oportunidades BNF.

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
    DATA_CORTE = datetime.strptime(RAW_DATA, "%Y-%m-%d")
    DATA_INICIO = DATA_CORTE - timedelta(days=400)  # ~1.5 anos de histórico

    if DATA_CORTE > datetime.now():
        raise ValueError("A data de corte não pode ser futura!")

    print(f"✅ Data de corte válida: {DATA_CORTE.strftime('%d/%m/%Y')}")
    print(f"📆 Período de análise: {DATA_INICIO.strftime('%d/%m/%Y')} → {DATA_CORTE.strftime('%d/%m/%Y')}")
    print(f"🕐 {(DATA_CORTE - DATA_INICIO).days} dias de histórico carregados")

except ValueError as e:
    raise ValueError(f"❌ Data inválida: {e}. Use o formato YYYY-MM-DD (ex: 2024-01-15)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Lista de Ações da B3
# MAGIC
# MAGIC O notebook usa a lista das principais ações negociadas na B3 (Bolsa de Valores do Brasil),
# MAGIC cobrindo os principais índices: IBOV, IBRX-100, SMLL, IDIV.

# COMMAND ----------

# DBTITLE 1,Lista Completa de Tickers B3
# Lista curada das principais acoes da B3 (~130 ativos)
# Todos os tickers terminam em .SA para o Yahoo Finance
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
    # Referencia de indice
    "^BVSP",  # IBOVESPA
]

print(f"📋 Total de ativos na lista: {len(TICKERS_B3) - 1} acoes + 1 indice (IBOV)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔽 Download de Dados Históricos

# COMMAND ----------

# DBTITLE 1,Funcoes de Download e Indicadores Tecnicos
def baixar_dados(tickers, inicio, fim, batch_size=15):
    """
    Baixa dados OHLCV em lotes para evitar sobrecarga na API do Yahoo Finance.
    Retorna um dicionario {ticker: DataFrame}.
    """
    dados = {}
    fim_str    = fim.strftime("%Y-%m-%d")
    inicio_str = inicio.strftime("%Y-%m-%d")
    batches    = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    for idx, batch in enumerate(batches):
        print(f"  📦 Lote {idx+1}/{len(batches)}: baixando {len(batch)} ativos...")
        try:
            raw = yf.download(
                batch,
                start=inicio_str,
                end=fim_str,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df = raw.xs(tk, axis=1, level=1).copy()
                        df.dropna(subset=["Close"], inplace=True)
                        if len(df) >= 40:
                            dados[tk] = df
                    except Exception:
                        pass
            else:
                tk = batch[0]
                raw.dropna(subset=["Close"], inplace=True)
                if len(raw) >= 40:
                    dados[tk] = raw

        except Exception as e:
            print(f"    ⚠️ Erro no lote: {e}")

        time.sleep(0.4)

    return dados


def calcular_indicadores(df):
    """
    Calcula todos os indicadores tecnicos usados pelas estrategias BNF.
    """
    d = df.copy()

    # Medias Moveis
    d["MM9"]   = d["Close"].rolling(9).mean()
    d["MM21"]  = d["Close"].rolling(21).mean()
    d["MM25"]  = d["Close"].rolling(25).mean()
    d["MM55"]  = d["Close"].rolling(55).mean()
    d["MM200"] = d["Close"].rolling(200).mean()

    # Kairi Ritsu — desvio % em relacao a MM25 (indicador principal do BNF)
    d["KAIRI_25"] = ((d["Close"] - d["MM25"]) / d["MM25"]) * 100

    # RSI (Wilder, 14 periodos)
    delta    = d["Close"].diff()
    ganho    = delta.clip(lower=0)
    perda    = (-delta).clip(lower=0)
    rsi_gain = ganho.ewm(com=13, adjust=False).mean()
    rsi_loss = perda.ewm(com=13, adjust=False).mean()
    rs       = rsi_gain / rsi_loss.replace(0, np.nan)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR — Average True Range
    hl   = d["High"] - d["Low"]
    hcp  = (d["High"] - d["Close"].shift(1)).abs()
    lcp  = (d["Low"]  - d["Close"].shift(1)).abs()
    tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d["ATR_14"]  = tr.rolling(14).mean()
    d["ATR_PCT"] = (d["ATR_14"] / d["Close"]) * 100

    # Bollinger Bands (20, 2σ)
    bb_mid         = d["Close"].rolling(20).mean()
    bb_std         = d["Close"].rolling(20).std()
    d["BB_UPPER"]  = bb_mid + 2 * bb_std
    d["BB_LOWER"]  = bb_mid - 2 * bb_std
    d["BB_WIDTH"]  = ((d["BB_UPPER"] - d["BB_LOWER"]) / bb_mid) * 100

    # Volume Ratio
    vol_ma         = d["Volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["Volume"] / vol_ma.replace(0, np.nan)

    # Retornos
    d["RETORNO_5D"]  = d["Close"].pct_change(5) * 100
    d["RETORNO_20D"] = d["Close"].pct_change(20) * 100
    d["RETORNO_60D"] = d["Close"].pct_change(60) * 100

    # 52 semanas
    d["MAX_52W"] = d["Close"].rolling(252).max()
    d["MIN_52W"] = d["Close"].rolling(252).min()
    d["DIST_MAX"] = ((d["Close"] - d["MAX_52W"]) / d["MAX_52W"]) * 100
    d["DIST_MIN"] = ((d["Close"] - d["MIN_52W"]) / d["MIN_52W"]) * 100

    return d


print("✅ Funcoes definidas.")

# COMMAND ----------

# DBTITLE 1,Download dos Dados
print("🔽 Iniciando download de dados historicos...")
print(f"   Periodo: {DATA_INICIO.strftime('%d/%m/%Y')} → {DATA_CORTE.strftime('%d/%m/%Y')}\n")

DADOS_BRUTOS = baixar_dados(TICKERS_B3, DATA_INICIO, DATA_CORTE)

DADOS_IBOV  = DADOS_BRUTOS.pop("^BVSP", None)
DADOS_ACOES = dict(DADOS_BRUTOS)

print(f"\n✅ Download concluido!")
print(f"   📊 Acoes com dados validos: {len(DADOS_ACOES)}")
if DADOS_IBOV is not None:
    print(f"   📈 IBOV: {len(DADOS_IBOV)} pregoes carregados")
else:
    print("   ⚠️ IBOV nao disponivel — regime sera estimado pelas proprias acoes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Calculando Indicadores Técnicos

# COMMAND ----------

# DBTITLE 1,Aplicacao dos Indicadores
print("📐 Calculando indicadores para todos os ativos...\n")

DADOS_COM_IND = {}
for ticker, df in DADOS_ACOES.items():
    try:
        DADOS_COM_IND[ticker] = calcular_indicadores(df)
    except Exception:
        pass

if DADOS_IBOV is not None:
    DADOS_IBOV = calcular_indicadores(DADOS_IBOV)

print(f"✅ Indicadores calculados para {len(DADOS_COM_IND)} ativos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌡️ Detecção do Regime de Mercado (Bear vs. Bull)
# MAGIC
# MAGIC Antes de aplicar as estrategias, precisamos saber o **estado do mercado** na data de corte.
# MAGIC
# MAGIC BNF entendia que as estratégias mudam conforme o ciclo do mercado:
# MAGIC - **Bear Market** → Pánico + oversold extremo = campo fértil para reversão à média
# MAGIC - **Bull Market** → Ações atrasadas que não subiram com o índice = candidatas a "pegar o trem"
# MAGIC
# MAGIC ### Critérios para classificação:
# MAGIC | Condição | Bear | Neutro | Bull |
# MAGIC |----------|------|--------|------|
# MAGIC | IBOV vs MM200 | Abaixo | Próximo | Acima |
# MAGIC | Retorno 60d | < -10% | -10% a +10% | > +10% |
# MAGIC | RSI IBOV | < 40 | 40–60 | > 60 |

# COMMAND ----------

# DBTITLE 1,Classificacao do Regime de Mercado
def detectar_regime(ibov_df):
    if ibov_df is None or len(ibov_df) < 50:
        return "NEUTRO", "Dados insuficientes do IBOV — regime indeterminado"

    ult = ibov_df.iloc[-1]
    acima_mm200 = ult["Close"] > ult.get("MM200", ult["Close"])
    ret60       = ult.get("RETORNO_60D", 0)
    rsi         = ult.get("RSI_14", 50)
    dist_max    = ult.get("DIST_MAX", 0)

    pontos_bull = pontos_bear = 0
    if acima_mm200:        pontos_bull += 1
    else:                  pontos_bear += 1
    if ret60 > 10:         pontos_bull += 1
    elif ret60 < -10:      pontos_bear += 1
    if rsi > 55:           pontos_bull += 1
    elif rsi < 45:         pontos_bear += 1
    if dist_max > -10:     pontos_bull += 1
    elif dist_max < -25:   pontos_bear += 1

    if pontos_bull >= 3:
        return "BULL", f"IBOV em alta tendencia (RSI={rsi:.1f}, Ret60d={ret60:.1f}%, {'acima' if acima_mm200 else 'abaixo'} da MM200)"
    elif pontos_bear >= 3:
        return "BEAR", f"IBOV em queda/panico (RSI={rsi:.1f}, Ret60d={ret60:.1f}%, {'acima' if acima_mm200 else 'abaixo'} da MM200)"
    else:
        return "NEUTRO", f"IBOV em consolidacao lateral (RSI={rsi:.1f}, Ret60d={ret60:.1f}%)"


REGIME, REGIME_DESC = detectar_regime(DADOS_IBOV)
emoji_r = {"BULL": "🟢", "BEAR": "🔴", "NEUTRO": "🟡"}
print(f"\n{emoji_r.get(REGIME,'⚪')} REGIME DETECTADO: {REGIME}")
print(f"   {REGIME_DESC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Estratégias BNF — Implementação

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 1: Reversão à Média via Kairi (MM25)
# MAGIC
# MAGIC **O que é?**
# MAGIC BNF comprava ações quando o **Kairi** (desvio do preço em relação à MM25) atingia níveis extremamente negativos.
# MAGIC Ele entendia que o mercado frequentemente reage em excesso.
# MAGIC
# MAGIC **Regras:**
# MAGIC - Kairi < -20%: Oversold severo → candidato a reversão
# MAGIC - RSI < 40 = oversold confirmado
# MAGIC - Volume acima da média = pânico real (não fuga silenciosa)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC Quando uma ação cai 20-35% sem mudança fundamental, é geralmente pânico de curto prazo.
# MAGIC Os compradores "racionais" entram e o preço reverte naturalmente em direção à média.

# COMMAND ----------

# DBTITLE 1,Estrategia 1 — Reversao a Media (Kairi)
def estrategia_reversao_media(df, ticker):
    if len(df) < 30:
        return None
    ult      = df.iloc[-1]
    kairi    = ult.get("KAIRI_25", 0)
    rsi      = ult.get("RSI_14", 50)
    vol_rat  = ult.get("VOL_RATIO", 1.0)
    ret5     = ult.get("RETORNO_5D", 0)

    if not (kairi < -20 and rsi < 40):
        return None

    score = 0
    if kairi < -30:    score += 40
    elif kairi < -25:  score += 25
    else:              score += 15
    if rsi < 25:       score += 25
    elif rsi < 30:     score += 15
    else:              score += 10
    if vol_rat > 1.2:  score += 20
    if ret5 < -10:     score += 15

    mm25 = ult.get("MM25", ult["Close"] * 1.20)
    alvo = mm25
    pot  = ((alvo - ult["Close"]) / ult["Close"]) * 100

    return {
        "ticker": ticker, "estrategia": "1 — Reversão à Média (Kairi)",
        "score": min(score, 100), "preco": round(ult["Close"], 2),
        "kairi": round(kairi, 2), "rsi": round(rsi, 2),
        "vol_ratio": round(vol_rat, 2), "ret20d": round(ult.get("RETORNO_20D", 0), 2),
        "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"O ativo está {abs(kairi):.1f}% ABAIXO da MM25 (Kairi={kairi:.1f}%), "
            f"com RSI={rsi:.1f} — oversold confirmado. "
            f"BNF compraria aqui apostando na reversão ao nível da MM25 "
            f"(alvo: R$ {alvo:.2f}, potencial de +{pot:.1f}%). "
            f"Volume {vol_rat:.1f}x a média {'confirma pânico real.' if vol_rat > 1.2 else '(baixo — atenção).'}"
        ),
        "regime_alvo": ["BEAR", "NEUTRO", "BULL"],
    }

print("✅ Estrategia 1 definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 2: Pânico com Alta Volatilidade (Bear Market)
# MAGIC
# MAGIC **O que é?**
# MAGIC Em mercados de baixa, BNF se especializava em ações com **volatilidade extremamente alta** sendo
# MAGIC vendidas de forma irracional. Alta volatilidade = maior amplitude = maior lucro por operação.
# MAGIC
# MAGIC **Regras:**
# MAGIC - ATR% > 4% ao dia (altamente volátil)
# MAGIC - Preço tocando ou abaixo da banda inferior de Bollinger
# MAGIC - RSI < 35 + queda de pelo menos -15% nos últimos 20 dias
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC Alta volatilidade em bear market = o pêndulo oscila violentamente para os dois lados.
# MAGIC BNF entrava na baixa extrema e saía no primeiro soluço de alta, capturando 5-15% em horas.

# COMMAND ----------

# DBTITLE 1,Estrategia 2 — Volatilidade + Panico (Bear Market)
def estrategia_volatilidade_bear(df, ticker, regime):
    if len(df) < 30 or regime == "BULL":
        return None
    ult      = df.iloc[-1]
    atr_pct  = ult.get("ATR_PCT", 0)
    bb_lower = ult.get("BB_LOWER", 0)
    bb_width = ult.get("BB_WIDTH", 0)
    rsi      = ult.get("RSI_14", 50)
    ret20    = ult.get("RETORNO_20D", 0)
    close    = ult["Close"]

    if not (atr_pct > 4.0 and rsi < 38 and ret20 < -10):
        return None

    score = 0
    if atr_pct > 8:    score += 35
    elif atr_pct > 6:  score += 25
    else:              score += 15
    if close <= bb_lower * 1.02:  score += 25
    if rsi < 25:       score += 20
    elif rsi < 30:     score += 12
    if ret20 < -25:    score += 20
    elif ret20 < -15:  score += 12
    if bb_width > 30:  score += 10

    mm25 = ult.get("MM25", close * 1.10)
    alvo = close + (mm25 - close) * 0.5
    pot  = ((alvo - close) / close) * 100

    return {
        "ticker": ticker, "estrategia": "2 — Volatilidade + Pânico (Bear)",
        "score": min(score, 100), "preco": round(close, 2),
        "atr_pct": round(atr_pct, 2), "bb_width": round(bb_width, 2),
        "rsi": round(rsi, 2), "ret20d": round(ret20, 2),
        "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Volatilidade diária de {atr_pct:.1f}% (ATR%), queda de {abs(ret20):.1f}% em 20 pregões e RSI={rsi:.1f}. "
            f"BNF adorava esse cenário de bear market: ações caindo violentamente com volatilidade extrema. "
            f"Operação de SCALP rápido — entrar na baixa extrema e sair no primeiro respiro de alta "
            f"(alvo: R$ {alvo:.2f}, +{pot:.1f}%). STOP CURTO é obrigatório."
        ),
        "regime_alvo": ["BEAR", "NEUTRO"],
    }

print("✅ Estrategia 2 definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 3: Atrasados no Rally (Bull Market)
# MAGIC
# MAGIC **O que é?**
# MAGIC Em mercados de alta, BNF observava que **nem todas as ações sobem ao mesmo tempo**.
# MAGIC Quando o IBOV sobe consistentemente, algumas ações ficam "para trás".
# MAGIC Eventualmente, o dinheiro "rota" para esses ativos — eles fazem um movimento de catch-up.
# MAGIC
# MAGIC **Regras:**
# MAGIC - IBOV subiu > +8% nos últimos 60 pregões (bull market confirmado)
# MAGIC - A ação subiu menos que 50% do IBOV (ficou para trás)
# MAGIC - RSI entre 35-65 (interesse comprador começando, mas não overbought)
# MAGIC - Preço acima da MM55 (não está em colapso)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC O capital institucional vai setor por setor. Identificar quais ações ainda não foram
# MAGIC "visitadas" permite se posicionar ANTES do dinheiro grande chegar.

# COMMAND ----------

# DBTITLE 1,Estrategia 3 — Atrasados no Rally (Bull Market)
def estrategia_atraso_rally(df, ticker, ibov_df, regime):
    if len(df) < 65 or regime != "BULL":
        return None
    ult         = df.iloc[-1]
    ret60_acao  = ult.get("RETORNO_60D", 0)
    rsi         = ult.get("RSI_14", 50)
    close       = ult["Close"]
    mm55        = ult.get("MM55", 0)
    mm9         = ult.get("MM9", 0)
    mm21        = ult.get("MM21", 0)
    vol_ratio   = ult.get("VOL_RATIO", 1.0)

    ret60_ibov = 0
    if ibov_df is not None and len(ibov_df) >= 60:
        ret60_ibov = ibov_df.iloc[-1].get("RETORNO_60D", 0)

    if ret60_ibov < 8:
        return None

    lag_ratio   = ret60_acao / ret60_ibov if ret60_ibov != 0 else 1
    if lag_ratio >= 0.5:
        return None
    if mm55 > 0 and close < mm55 * 0.95:
        return None
    if not (35 <= rsi <= 65):
        return None

    score = 0
    score += min(int((0.5 - lag_ratio) * 100), 40)
    if ret60_ibov > 20:    score += 20
    elif ret60_ibov > 15:  score += 12
    if vol_ratio > 1.3:    score += 15
    if mm9 > mm21:         score += 15
    if rsi > 50:           score += 10

    alvo_ret = ret60_ibov * 0.75
    alvo     = close * (1 + (alvo_ret - ret60_acao) / 100)
    pot      = ((alvo - close) / close) * 100

    return {
        "ticker": ticker, "estrategia": "3 — Atrasados no Rally (Bull)",
        "score": min(score, 100), "preco": round(close, 2),
        "ret60_acao": round(ret60_acao, 2), "ret60_ibov": round(ret60_ibov, 2),
        "lag_ratio": round(lag_ratio, 2), "rsi": round(rsi, 2),
        "vol_ratio": round(vol_ratio, 2), "alvo": round(alvo, 2),
        "potencial": round(pot, 2),
        "explicacao": (
            f"O IBOV subiu {ret60_ibov:.1f}% nos últimos 60 pregões, mas {ticker.replace('.SA','')} subiu apenas "
            f"{ret60_acao:.1f}% — ficou {(1-lag_ratio)*100:.0f}% para trás do índice. "
            f"BNF chamava isso de 'pegar o bonde atrasado': quando o dinheiro rotaciona, "
            f"esses ativos costumam ter movimentos bruscos de catch-up. "
            f"RSI={rsi:.1f} indica que o interesse comprador está apenas começando. "
            f"Alvo: R$ {alvo:.2f} (+{pot:.1f}% de potencial)."
        ),
        "regime_alvo": ["BULL"],
    }

print("✅ Estrategia 3 definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 4: Exaustão de Volume (Vendedores Esgotados)
# MAGIC
# MAGIC **O que é?**
# MAGIC BNF monitorava o volume com atenção cirúrgica. Quando uma ação caía em **volume alto** (pânico)
# MAGIC e então o volume **secava** enquanto o preço parava de cair, ele interpretava como
# MAGIC **esgotamento dos vendedores**. Sem mais vendedores, qualquer compra move o preço.
# MAGIC
# MAGIC **Regras:**
# MAGIC - Queda ≥ -15% nos últimos 20 dias
# MAGIC - Volume dos últimos 5 dias < 60% da média de 20 dias (volume secou)
# MAGIC - Preço parou de cair (variação dos últimos 5 dias < ±5%)
# MAGIC - Fechamento próximo da máxima do dia (pressão compradora emergindo)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC Depois de uma queda longa, os "fracos" já venderam tudo. Quem sobrou são os "holdlers" convictos.
# MAGIC Quando o volume seca, o fluxo vendedor se esgota — qualquer compra inicia a reversão.

# COMMAND ----------

# DBTITLE 1,Estrategia 4 — Exaustao de Volume
def estrategia_exaustao_volume(df, ticker):
    if len(df) < 25:
        return None
    ult   = df.iloc[-1]
    ult5  = df.iloc[-5:]
    close = ult["Close"]
    ret20 = ult.get("RETORNO_20D", 0)
    rsi   = ult.get("RSI_14", 50)

    vol_5d  = ult5["Volume"].mean() if "Volume" in ult5.columns else 0
    vol_20d = df["Volume"].iloc[-20:].mean() if "Volume" in df.columns else 1
    vol_rat = vol_5d / vol_20d if vol_20d > 0 else 1

    if len(ult5) >= 5:
        ret5r = ((ult5["Close"].iloc[-1] - ult5["Close"].iloc[0]) / ult5["Close"].iloc[0]) * 100
    else:
        return None

    rng_dia  = ult["High"] - ult["Low"]
    pos_fech = ((close - ult["Low"]) / rng_dia) if rng_dia > 0 else 0.5

    if not (ret20 < -15 and vol_rat < 0.60):
        return None

    score = 0
    if ret20 < -25:    score += 30
    else:              score += 20
    if vol_rat < 0.40: score += 30
    elif vol_rat < 0.50: score += 20
    else:              score += 10
    if abs(ret5r) < 5: score += 20
    if pos_fech > 0.6: score += 15
    if rsi < 35:       score += 5

    mm25 = ult.get("MM25", close * 1.15)
    alvo = close + (mm25 - close) * 0.5
    pot  = ((alvo - close) / close) * 100

    return {
        "ticker": ticker, "estrategia": "4 — Exaustão de Volume",
        "score": min(score, 100), "preco": round(close, 2),
        "vol_ratio": round(vol_rat, 2), "ret20d": round(ret20, 2),
        "rsi": round(rsi, 2), "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Após queda de {abs(ret20):.1f}% em 20 pregões, o volume secou para "
            f"{vol_rat*100:.0f}% da média — os vendedores se esgotaram. "
            f"Preço estabilizando (±{abs(ret5r):.1f}% nos últimos 5 dias), "
            f"fechamento no {pos_fech*100:.0f}% superior do candle = pressão compradora emergindo. "
            f"BNF via isso como o 'piso do desespero'. Alvo: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BEAR", "NEUTRO", "BULL"],
    }

print("✅ Estrategia 4 definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📘 ESTRATÉGIA 5: Sniper de Baixo Risco (Setup Técnico Perfeito)
# MAGIC
# MAGIC **O que é?**
# MAGIC BNF descrevia-se como um **atirador de elite**: espera, espera, espera... e dispara apenas
# MAGIC quando o alvo está perfeito. Esta estratégia busca o setup com múltiplas confirmações alinhadas.
# MAGIC
# MAGIC **Regras:**
# MAGIC - RSI entre 42-62 (saindo do oversold, não overbought)
# MAGIC - MM9 cruzando MM21 para cima (cruzamento dourado de curto prazo)
# MAGIC - Volume acima da média (confirma o movimento)
# MAGIC - Preço acima da MM55 (tendência de médio prazo intacta)
# MAGIC - Kairi entre -15% e +5% (preço normalizado)
# MAGIC
# MAGIC **Por que funciona?**
# MAGIC Múltiplas confirmações reduzem os falsos sinais.
# MAGIC BNF preferia perder alguns trades aguardando o setup perfeito a entrar em setups mediocres.
# MAGIC **A paciência era sua maior arma.**

# COMMAND ----------

# DBTITLE 1,Estrategia 5 — Sniper de Baixo Risco
def estrategia_sniper(df, ticker):
    if len(df) < 60:
        return None
    ult  = df.iloc[-1]
    ant3 = df.iloc[-4:-1]

    close   = ult["Close"]
    mm9     = ult.get("MM9", 0)
    mm21    = ult.get("MM21", 0)
    mm55    = ult.get("MM55", 0)
    rsi     = ult.get("RSI_14", 50)
    vol_rat = ult.get("VOL_RATIO", 1.0)
    kairi   = ult.get("KAIRI_25", 0)

    cruzamento = False
    if len(ant3) >= 2 and "MM9" in ant3.columns and "MM21" in ant3.columns:
        for i in range(len(ant3) - 1):
            m9b = ant3["MM9"].iloc[i]; m21b = ant3["MM21"].iloc[i]
            m9d = ant3["MM9"].iloc[i+1]; m21d = ant3["MM21"].iloc[i+1]
            if not (pd.isna(m9b) or pd.isna(m21b)):
                if m9b <= m21b and m9d > m21d:
                    cruzamento = True
                    break

    mm9_up = mm9 > mm21 * 1.001 if mm9 > 0 and mm21 > 0 else False
    ok_cruz = cruzamento or (mm9_up and rsi > 48)
    ok_rsi  = 42 <= rsi <= 62
    ok_mm55 = close > mm55 * 0.98 if mm55 > 0 else True
    ok_kai  = -15 <= kairi <= 5

    if not (ok_cruz and ok_rsi and ok_mm55 and ok_kai):
        return None

    score = 0
    if cruzamento:        score += 35
    else:                 score += 15
    if 48 <= rsi <= 55:   score += 25
    elif ok_rsi:          score += 15
    if vol_rat >= 1.1:    score += 20
    if ok_mm55:           score += 10
    if -5 <= kairi <= 0:  score += 10

    alvo = close * 1.08
    pot  = 8.0

    return {
        "ticker": ticker, "estrategia": "5 — Sniper de Baixo Risco",
        "score": min(score, 100), "preco": round(close, 2),
        "kairi": round(kairi, 2), "rsi": round(rsi, 2),
        "vol_ratio": round(vol_rat, 2), "alvo": round(alvo, 2), "potencial": round(pot, 2),
        "explicacao": (
            f"Setup Sniper: {'⚡ MM9 cruzou MM21 recentemente' if cruzamento else 'MM9 > MM21 com momentum positivo'}, "
            f"RSI={rsi:.1f} (zona ideal 42-62), "
            f"volume {vol_rat:.1f}x acima da média confirmando, "
            f"preço {'acima' if close > mm55 else 'próximo'} da MM55 (tendência OK), "
            f"Kairi={kairi:.1f}% (preço normalizado). "
            f"Múltiplas confirmações reduzem risco de entrada falsa. "
            f"Alvo conservador: R$ {alvo:.2f} (+{pot:.1f}%)."
        ),
        "regime_alvo": ["BULL", "NEUTRO", "BEAR"],
    }

print("✅ Estrategia 5 definida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Execução de Todas as Estratégias

# COMMAND ----------

# DBTITLE 1,Varredura Completa
print("🔍 Iniciando varredura de estrategias BNF...\n")

resultados_todos = []
for ticker, df in DADOS_COM_IND.items():
    for fn in [
        lambda d, t: estrategia_reversao_media(d, t),
        lambda d, t: estrategia_volatilidade_bear(d, t, REGIME),
        lambda d, t: estrategia_atraso_rally(d, t, DADOS_IBOV, REGIME),
        lambda d, t: estrategia_exaustao_volume(d, t),
        lambda d, t: estrategia_sniper(d, t),
    ]:
        try:
            r = fn(df, ticker)
            if r:
                resultados_todos.append(r)
        except Exception:
            pass

print(f"✅ Varredura concluída! Oportunidades encontradas: {len(resultados_todos)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Resultados e Recomendações

# COMMAND ----------

# DBTITLE 1,Exibicao HTML dos Resultados
resultados_filtrados = sorted(
    [r for r in resultados_todos if REGIME in r.get("regime_alvo", [REGIME])],
    key=lambda x: x["score"],
    reverse=True,
)

ECORES = {"1": "#FF6B6B", "2": "#FF4444", "3": "#00C851", "4": "#FFBB33", "5": "#33B5E5"}
EICONS = {"1": "📉", "2": "🌩️", "3": "🚀", "4": "💤", "5": "🎯"}
RCORES = {"BULL": "#00C851", "BEAR": "#FF4444", "NEUTRO": "#FFBB33"}

def score_badge(score):
    cor = "#00C851" if score >= 75 else ("#FFBB33" if score >= 55 else "#FF6B6B")
    return f'<span style="background:{cor};color:#000;padding:2px 10px;border-radius:12px;font-weight:800;">{score}</span>'

def gerar_html(resultados, regime, data_corte):
    rcor = RCORES.get(regime, "#888")
    rics = {"BULL": "🟢 BULL", "BEAR": "🔴 BEAR", "NEUTRO": "🟡 NEUTRO"}

    if not resultados:
        return f"""
        <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
             padding:32px;border-radius:16px;border:2px solid #FF4444;">
          <h2 style="color:#FF4444;margin-top:0;">⚠️ Nenhum Ativo Encontrado com Padrão BNF</h2>
          <p style="font-size:16px;">
            Na data <strong>{data_corte.strftime('%d/%m/%Y')}</strong> com regime
            <strong style="color:{rcor}">{rics.get(regime, regime)}</strong>,
            nenhuma ação brasileira se enquadrou nas estratégias de Takashi Kotegawa (BNF).
          </p>
          <hr style="border-color:#45475A;">
          <p style="color:#BAC2DE;font-size:14px;">
            💡 <strong>O que isso significa?</strong><br>
            BNF era extremamente seletivo — chamava seu estilo de "sniper".
            Preferia não operar a entrar em setups mediocres.<br>
            Se não há oportunidade clara nessa data, a decisão mais sábia é <strong>não operar</strong>
            e aguardar condições mais favoráveis. Tente outras datas ou períodos de maior volatilidade.
          </p>
        </div>"""

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
         padding:24px;border-radius:16px;">
      <h1 style="color:#CBA6F7;margin:0 0 20px 0;">🎯 BNF Simulator — Oportunidades Detectadas</h1>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid #CBA6F7;">
          📅 <b>Data:</b> {data_corte.strftime('%d/%m/%Y')}
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid {rcor};">
          🌡️ <b>Regime:</b> <span style="color:{rcor}">{rics.get(regime, regime)}</span>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 18px;border-left:4px solid #89B4FA;">
          📊 <b>Oportunidades:</b> {len(resultados)} ativos
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
            ({len(items)} ativo{'s' if len(items)>1 else ''})
          </span>
        </h2>"""

        for r in items[:10]:
            tk  = r["ticker"].replace(".SA", "")
            pc  = r.get("preco", 0)
            alv = r.get("alvo", 0)
            pot = r.get("potencial", 0)
            pc2 = "#00C851" if pot > 0 else "#FF4444"

            mex = ""
            for k, lbl in [("kairi","Kairi"),("rsi","RSI"),("vol_ratio","Vol"),("atr_pct","ATR%"),
                            ("ret60_acao","Ret60d"),("ret60_ibov","IBOV60d"),("bb_width","BB-Width")]:
                if k in r:
                    v = r[k]
                    suffix = "%" if k in ["kairi","ret60_acao","ret60_ibov","atr_pct","bb_width"] else "x" if k=="vol_ratio" else ""
                    mex += f'<span style="background:#45475A;padding:2px 8px;border-radius:8px;margin-right:5px;font-size:12px;">{lbl}: <b>{v}{suffix}</b></span>'

            html += f"""
            <div style="background:#313244;border-radius:12px;padding:14px;margin-bottom:12px;border-left:4px solid {cor};">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                  <span style="font-size:21px;font-weight:800;">{tk}</span>
                  <span style="margin-left:10px;color:#888;font-size:12px;">{r['estrategia']}</span>
                </div>
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                  <span>Score: {score_badge(r['score'])}</span>
                  <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;">
                    💰 R$ <b>{pc:.2f}</b>
                  </span>
                  <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;">
                    🎯 <b style="color:{pc2}">R$ {alv:.2f}</b>
                    <em style="color:{pc2}"> (+{pot:.1f}%)</em>
                  </span>
                </div>
              </div>
              <div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:5px;">{mex}</div>
              <div style="background:#1E1E2E;border-radius:8px;padding:10px;font-size:13px;
                   color:#BAC2DE;line-height:1.6;">
                🧠 <b style="color:{cor}">Por que BNF operaria?</b><br>{r['explicacao']}
              </div>
            </div>"""

    html += """
      <hr style="border-color:#45475A;margin-top:20px;">
      <div style="background:#313244;border-radius:10px;padding:14px;font-size:12px;color:#888;">
        ⚠️ <b>Disclaimer:</b> Este notebook é puramente educacional e não constitui recomendação de investimento.
        As estratégias de BNF foram desenvolvidas para o mercado japonês dos anos 2000.
        Sempre utilize stop-loss e opere apenas capital que pode perder.
      </div>
    </div>"""
    return html


displayHTML(gerar_html(resultados_filtrados, REGIME, DATA_CORTE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Tabela Resumo

# COMMAND ----------

# DBTITLE 1,Tabela de Resultados
if resultados_filtrados:
    COLS_ORDER = ["ticker", "estrategia", "score", "preco", "alvo", "potencial",
                  "rsi", "kairi", "vol_ratio", "ret20d", "atr_pct",
                  "ret60_acao", "ret60_ibov", "lag_ratio"]
    df_res = pd.DataFrame(resultados_filtrados)
    cols   = [c for c in COLS_ORDER if c in df_res.columns]
    df_ex  = df_res[cols].copy()
    df_ex["ticker"] = df_ex["ticker"].str.replace(".SA", "", regex=False)
    df_ex  = df_ex.sort_values(["estrategia", "score"], ascending=[True, False]).reset_index(drop=True)

    print(f"📋 {len(df_ex)} oportunidades | {DATA_CORTE.strftime('%d/%m/%Y')} | Regime: {REGIME}\n")
    display(df_ex)

    try:
        caminho = f"/tmp/bnf_resultados_{DATA_CORTE.strftime('%Y%m%d')}.csv"
        df_ex.to_csv(caminho, index=False)
        print(f"\n💾 CSV salvo: {caminho}")
    except Exception as e:
        print(f"\n⚠️ Nao foi possivel salvar CSV: {e}")
else:
    print(f"⚠️ Nenhuma oportunidade BNF para {DATA_CORTE.strftime('%d/%m/%Y')} ({REGIME})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Referência das Estratégias BNF
# MAGIC
# MAGIC | Estratégia | Regime | Indicadores Principais | Critério de Entrada |
# MAGIC |-----------|--------|------------------------|---------------------|
# MAGIC | 1 — Reversão à Média | Any | Kairi, MM25, RSI | Kairi < -20%, RSI < 40 |
# MAGIC | 2 — Volatilidade Bear | Bear/Neutro | ATR%, Bollinger, RSI | ATR > 4%, BB inferior tocada |
# MAGIC | 3 — Atrasados no Rally | Bull | Ret60d vs IBOV | Ação subiu < 50% do IBOV |
# MAGIC | 4 — Exaustão de Volume | Any | Volume Ratio, Ret20d | Vol < 60% média + queda > 15% |
# MAGIC | 5 — Sniper | Any | MM9/MM21, RSI, Vol | Cruzamento dourado + RSI neutro |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏆 Os 7 Princípios Imutáveis de BNF
# MAGIC
# MAGIC 1. **Nunca opere sem stop-loss** — preservar capital é prioridade absoluta
# MAGIC 2. **Liquidez é obrigatória** — só opere ativos com volume suficiente para sair
# MAGIC 3. **O mercado está sempre certo** — se o setup falhar, saia sem hesitar
# MAGIC 4. **Paciência é vantagem** — espere o setup perfeito, não force entradas
# MAGIC 5. **Controle emocional** — pânico e ganância são seus piores inimigos
# MAGIC 6. **Diversifique as posições** — nunca coloque tudo em um único ativo
# MAGIC 7. **Kairi é seu guia** — o preço sempre reverte à média, mais cedo ou mais tarde

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📈 Backtesting — O que aconteceu depois?
# MAGIC
# MAGIC > Esta seção só é executada quando a **data de corte é no passado**.
# MAGIC > Se a data de corte for hoje ou futura, esta seção é ignorada automaticamente.
# MAGIC >
# MAGIC > Para cada ativo recomendado, verificamos:
# MAGIC > * ✅ Se o preço atingiu o **alvo** após a data de corte
# MAGIC > * ❌ Se o preço caiu abaixo de um **stop implícito** (-10% do preço de entrada)
# MAGIC > * 📅 Em qual data o alvo foi atingido (caso positivo)
# MAGIC > * 📊 Gráfico de preço do ativo do início ao fim, com linha vertical na data de corte

# COMMAND ----------

# DBTITLE 1,Verificação — Data de Corte no Passado?
import matplotlib
matplotlib.use("Agg")  # backend sem display (compatível com Databricks)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import base64
from io import BytesIO

HOJE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
DIAS_APOS_CORTE = (HOJE - DATA_CORTE).days

print(f"📅 Data de corte : {DATA_CORTE.strftime('%d/%m/%Y')}")
print(f"📅 Data atual    : {HOJE.strftime('%d/%m/%Y')}")
print(f"⏱️  Dias após corte: {DIAS_APOS_CORTE}")

if DIAS_APOS_CORTE < 3:
    print("\n🔵 Data de corte muito próxima de hoje — backtesting ignorado.")
    FAZER_BACKTEST = False
else:
    print(f"\n✅ {DIAS_APOS_CORTE} dias de dados disponíveis após o corte → executando backtesting!")
    FAZER_BACKTEST = True

# COMMAND ----------

# DBTITLE 1,Download de Dados Pós-Corte
if FAZER_BACKTEST and resultados_filtrados:
    tickers_rec = list({r["ticker"] for r in resultados_filtrados})
    print(f"🔽 Baixando dados pós-corte para {len(tickers_rec)} ativos recomendados...")
    print(f"   Período: {DATA_CORTE.strftime('%d/%m/%Y')} → {HOJE.strftime('%d/%m/%Y')}\n")

    # Download do período pós-corte (do dia seguinte ao corte até hoje)
    DATA_POS = DATA_CORTE + timedelta(days=1)
    DADOS_POS_CORTE = baixar_dados(
        tickers_rec,
        DATA_POS,
        HOJE + timedelta(days=1),  # +1 para incluir hoje
        batch_size=10,
    )

    # Download do período completo (início até hoje) para os gráficos
    print("\n🔽 Baixando histórico completo para gráficos...")
    DADOS_FULL = baixar_dados(
        tickers_rec,
        DATA_INICIO,
        HOJE + timedelta(days=1),
        batch_size=10,
    )

    print(f"\n✅ Dados pós-corte prontos para {len(DADOS_POS_CORTE)} ativos.")
else:
    DADOS_POS_CORTE = {}
    DADOS_FULL = {}
    if not resultados_filtrados:
        print("ℹ️ Sem recomendações — backtesting ignorado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔬 Análise de Desempenho das Recomendações
# MAGIC
# MAGIC ### Metodologia de avaliação:
# MAGIC | Resultado | Critério |
# MAGIC |-----------|----------|
# MAGIC | ✅ **Alvo atingido** | Preço máximo (High) cruzou o alvo em algum pregão pós-corte |
# MAGIC | 🛑 **Stop acionado** | Preço mínimo (Low) caiu abaixo de -10% do preço de entrada ANTES de atingir o alvo |
# MAGIC | ⏳ **Ainda aberto** | Nem alvo nem stop foram acionados até hoje |
# MAGIC | ❓ **Sem dados** | Dados insuficientes para avaliação |

# COMMAND ----------

# DBTITLE 1,Cálculo do Backtesting por Recomendação
STOP_PERCENTUAL = -0.10   # Stop implícito de -10%

def avaliar_recomendacao(ticker, preco_entrada, alvo, dados_pos):
    """
    Avalia se uma recomendação BNF funcionou após a data de corte.
    Retorna um dict com o resultado.
    """
    if ticker not in dados_pos or len(dados_pos[ticker]) == 0:
        return {
            "resultado": "❓ Sem dados",
            "data_resultado": None,
            "retorno_realizado": None,
            "max_preco": None,
            "min_preco": None,
            "dias_ate_alvo": None,
            "cor": "#888888",
        }

    df_pos = dados_pos[ticker].sort_index()
    stop_price = preco_entrada * (1 + STOP_PERCENTUAL)

    resultado = "⏳ Ainda aberto"
    data_res  = None
    dias_alvo = None
    cor       = "#FFBB33"

    for i, (dt, row) in enumerate(df_pos.iterrows()):
        high = row.get("High", row["Close"])
        low  = row.get("Low",  row["Close"])

        # Verifica stop primeiro (proteção de capital — BNF sempre priorizava isso)
        if low <= stop_price:
            resultado = f"🛑 Stop acionado (≤ R$ {stop_price:.2f})"
            data_res  = dt
            dias_alvo = i + 1
            cor       = "#FF4444"
            break

        # Verifica alvo
        if high >= alvo:
            resultado = f"✅ Alvo atingido (≥ R$ {alvo:.2f})"
            data_res  = dt
            dias_alvo = i + 1
            cor       = "#00C851"
            break

    # Retorno até o último pregão disponível (ou até o resultado)
    if data_res is not None:
        ultimo_preco = df_pos.loc[data_res]["Close"]
    else:
        ultimo_preco = df_pos.iloc[-1]["Close"]

    retorno_real = ((ultimo_preco - preco_entrada) / preco_entrada) * 100

    return {
        "resultado":         resultado,
        "data_resultado":    data_res.strftime("%d/%m/%Y") if data_res else "—",
        "retorno_realizado": round(retorno_real, 2),
        "ultimo_preco":      round(ultimo_preco, 2),
        "max_preco":         round(df_pos["High"].max(), 2) if "High" in df_pos.columns else None,
        "min_preco":         round(df_pos["Low"].min(), 2)  if "Low"  in df_pos.columns else None,
        "dias_ate_alvo":     dias_alvo,
        "cor":               cor,
    }


# Executa avaliação para todas as recomendações
resultados_backtest = []

if FAZER_BACKTEST and resultados_filtrados:
    print("🔬 Avaliando desempenho de cada recomendação...\n")
    for r in resultados_filtrados:
        av = avaliar_recomendacao(r["ticker"], r["preco"], r["alvo"], DADOS_POS_CORTE)
        resultados_backtest.append({
            **r,
            **av,
        })

    acertos  = sum(1 for x in resultados_backtest if "✅" in x["resultado"])
    stops    = sum(1 for x in resultados_backtest if "🛑" in x["resultado"])
    abertos  = sum(1 for x in resultados_backtest if "⏳" in x["resultado"])
    sem_dados= sum(1 for x in resultados_backtest if "❓" in x["resultado"])
    total    = len(resultados_backtest)

    print(f"✅ Alvos atingidos : {acertos}/{total} ({acertos/total*100:.0f}%)")
    print(f"🛑 Stops acionados : {stops}/{total}  ({stops/total*100:.0f}%)")
    print(f"⏳ Ainda abertos   : {abertos}/{total} ({abertos/total*100:.0f}%)")
    print(f"❓ Sem dados       : {sem_dados}/{total}")

# COMMAND ----------

# DBTITLE 1,Tabela HTML de Backtesting
def gerar_html_backtest(backtest, data_corte, hoje):
    if not backtest:
        return "<p style='color:#888;font-family:sans-serif;'>Sem dados de backtesting disponíveis.</p>"

    acertos = sum(1 for x in backtest if "✅" in x["resultado"])
    stops   = sum(1 for x in backtest if "🛑" in x["resultado"])
    abertos = sum(1 for x in backtest if "⏳" in x["resultado"])
    total   = len(backtest)
    taxa    = acertos / total * 100 if total > 0 else 0

    cor_taxa = "#00C851" if taxa >= 60 else ("#FFBB33" if taxa >= 40 else "#FF4444")

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;color:#CDD6F4;
         padding:24px;border-radius:16px;margin-top:16px;">
      <h2 style="color:#CBA6F7;margin:0 0 16px 0;">🔬 Backtesting — Desempenho das Recomendações BNF</h2>
      <p style="color:#888;font-size:13px;margin-bottom:16px;">
        Corte: <strong>{data_corte.strftime('%d/%m/%Y')}</strong> →
        Avaliado até: <strong>{hoje.strftime('%d/%m/%Y')}</strong> |
        Stop implícito: <strong>-10%</strong> do preço de entrada
      </p>

      <!-- Resumo -->
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid {cor_taxa};">
          🎯 <b>Taxa de acerto:</b> <span style="color:{cor_taxa};font-size:18px;font-weight:800;">{taxa:.0f}%</span>
          <span style="color:#888;font-size:12px;"> ({acertos}/{total})</span>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #00C851;">
          ✅ <b>Alvos atingidos:</b> <strong style="color:#00C851;">{acertos}</strong>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #FF4444;">
          🛑 <b>Stops acionados:</b> <strong style="color:#FF4444;">{stops}</strong>
        </div>
        <div style="background:#313244;border-radius:10px;padding:10px 20px;border-left:4px solid #FFBB33;">
          ⏳ <b>Ainda abertos:</b> <strong style="color:#FFBB33;">{abertos}</strong>
        </div>
      </div>

      <!-- Tabela -->
      <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#313244;color:#CBA6F7;">
            <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #45475A;">Ativo</th>
            <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #45475A;">Estratégia</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Entrada</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Alvo</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Stop</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Score</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Resultado</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Data Result.</th>
            <th style="padding:10px 12px;text-align:center;border-bottom:2px solid #45475A;">Dias</th>
            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #45475A;">Retorno Real</th>
          </tr>
        </thead>
        <tbody>
    """

    for i, r in enumerate(sorted(backtest, key=lambda x: x["score"], reverse=True)):
        tk       = r["ticker"].replace(".SA", "")
        entrada  = r.get("preco", 0)
        alvo_v   = r.get("alvo", 0)
        stop_v   = entrada * 0.90
        score    = r.get("score", 0)
        estrateg = r.get("estrategia", "")
        resultado= r.get("resultado", "❓")
        data_r   = r.get("data_resultado", "—")
        dias     = r.get("dias_ate_alvo", "—")
        ret_real = r.get("retorno_realizado", None)
        cor_linha= r.get("cor", "#888")
        bg       = "#2A2A3E" if i % 2 == 0 else "#252535"

        ret_str  = f"{ret_real:+.1f}%" if ret_real is not None else "—"
        ret_cor  = "#00C851" if (ret_real or 0) > 0 else "#FF4444"
        score_cor= "#00C851" if score >= 75 else ("#FFBB33" if score >= 55 else "#FF6B6B")

        html += f"""
          <tr style="background:{bg};border-left:3px solid {cor_linha};">
            <td style="padding:9px 12px;font-weight:700;color:#CDD6F4;">{tk}</td>
            <td style="padding:9px 12px;color:#888;font-size:12px;">{estrateg}</td>
            <td style="padding:9px 12px;text-align:right;">R$ {entrada:.2f}</td>
            <td style="padding:9px 12px;text-align:right;color:#00C851;">R$ {alvo_v:.2f}</td>
            <td style="padding:9px 12px;text-align:right;color:#FF4444;">R$ {stop_v:.2f}</td>
            <td style="padding:9px 12px;text-align:right;">
              <span style="background:{score_cor};color:#000;padding:1px 8px;border-radius:10px;font-weight:700;">{score}</span>
            </td>
            <td style="padding:9px 12px;text-align:center;font-weight:600;color:{cor_linha};">{resultado}</td>
            <td style="padding:9px 12px;text-align:center;color:#888;">{data_r}</td>
            <td style="padding:9px 12px;text-align:center;color:#888;">{dias if dias != "—" else "—"}</td>
            <td style="padding:9px 12px;text-align:right;font-weight:700;color:{ret_cor};">{ret_str}</td>
          </tr>"""

    html += """
        </tbody>
      </table>
      </div>
      <p style="color:#666;font-size:11px;margin-top:12px;">
        * Stop implícito de -10% calculado sobre o preço de entrada na data de corte.
        O stop real deve ser definido pelo trader conforme gestão de risco pessoal.
      </p>
    </div>"""
    return html


if FAZER_BACKTEST and resultados_backtest:
    displayHTML(gerar_html_backtest(resultados_backtest, DATA_CORTE, HOJE))
elif not FAZER_BACKTEST:
    print("🔵 Data de corte próxima de hoje — backtesting não aplicável.")

# COMMAND ----------

# DBTITLE 1,Tabela Pandas — Backtesting
if FAZER_BACKTEST and resultados_backtest:
    cols_bt = ["ticker", "estrategia", "score", "preco", "alvo", "potencial",
               "resultado", "data_resultado", "dias_ate_alvo",
               "retorno_realizado", "ultimo_preco", "max_preco", "min_preco"]
    df_bt = pd.DataFrame(resultados_backtest)
    cols_disp = [c for c in cols_bt if c in df_bt.columns]
    df_bt_ex = df_bt[cols_disp].copy()
    df_bt_ex["ticker"] = df_bt_ex["ticker"].str.replace(".SA", "", regex=False)
    df_bt_ex = df_bt_ex.sort_values("score", ascending=False).reset_index(drop=True)
    display(df_bt_ex)

    # Salva CSV
    try:
        path_bt = f"/tmp/bnf_backtest_{DATA_CORTE.strftime('%Y%m%d')}.csv"
        df_bt_ex.to_csv(path_bt, index=False)
        print(f"💾 CSV de backtest salvo: {path_bt}")
    except Exception as e:
        print(f"⚠️ Não foi possível salvar: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Gráficos de Preço — Ativo × Data de Corte
# MAGIC
# MAGIC Cada gráfico mostra:
# MAGIC - 📈 **Preço de fechamento** completo (histórico + pós-corte)
# MAGIC - 🟣 **Linha vertical** na data de corte (momento da recomendação)
# MAGIC - 🟢 **Linha horizontal verde** = preço alvo
# MAGIC - 🔴 **Linha horizontal vermelha** = preço de stop (-10%)
# MAGIC - ⚫ **Ponto laranja** = preço de entrada na data de corte

# COMMAND ----------

# DBTITLE 1,Geração dos Gráficos por Ativo
def plot_ativo_base64(ticker, dados_full, data_corte, preco_entrada, alvo, stop_price, resultado_info):
    """
    Gera o gráfico de preço de um ativo e retorna como imagem base64 para displayHTML.
    """
    if ticker not in dados_full or len(dados_full[ticker]) < 5:
        return None

    df_plot = dados_full[ticker].sort_index()

    # Divide em antes e depois da data de corte
    df_antes = df_plot[df_plot.index <= pd.Timestamp(data_corte)]
    df_depois = df_plot[df_plot.index >  pd.Timestamp(data_corte)]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Fundo escuro (estilo Databricks)
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    # Linha de preço — período antes do corte
    if len(df_antes) > 0:
        ax.plot(df_antes.index, df_antes["Close"],
                color="#89B4FA", linewidth=1.5, label="Preço (pré-corte)", zorder=3)

    # Linha de preço — período após o corte
    if len(df_depois) > 0:
        cor_pos = resultado_info.get("cor", "#CDD6F4")
        ax.plot(df_depois.index, df_depois["Close"],
                color=cor_pos, linewidth=2.0, label="Preço (pós-corte)", zorder=3)

    # Área sombreada pós-corte
    if len(df_depois) > 0:
        ax.axvspan(pd.Timestamp(data_corte), df_plot.index[-1],
                   alpha=0.08, color="#CBA6F7", zorder=1)

    # Linha vertical na data de corte
    ax.axvline(x=pd.Timestamp(data_corte), color="#CBA6F7",
               linewidth=2, linestyle="--", label=f"Data de corte ({data_corte.strftime('%d/%m/%Y')})", zorder=4)

    # Ponto de entrada (preço na data de corte)
    ax.scatter([pd.Timestamp(data_corte)], [preco_entrada],
               color="#FAB387", s=80, zorder=6, label=f"Entrada R$ {preco_entrada:.2f}")

    # Linha de alvo
    ax.axhline(y=alvo, color="#00C851", linewidth=1.2, linestyle="-.",
               label=f"Alvo R$ {alvo:.2f}", zorder=4)

    # Linha de stop
    ax.axhline(y=stop_price, color="#FF4444", linewidth=1.2, linestyle=":",
               label=f"Stop R$ {stop_price:.2f} (-10%)", zorder=4)

    # Marcação do resultado (se houver data de resultado)
    data_res_str = resultado_info.get("data_resultado", "—")
    if data_res_str and data_res_str != "—":
        try:
            data_res_dt = pd.Timestamp(datetime.strptime(data_res_str, "%d/%m/%Y"))
            if data_res_dt in df_plot.index or True:
                # Encontra o preço mais próximo nessa data
                idx_prox = df_plot.index.searchsorted(data_res_dt)
                if idx_prox < len(df_plot):
                    preco_res = df_plot["Close"].iloc[idx_prox]
                    cor_r = resultado_info.get("cor", "#FFBB33")
                    marker = "^" if "✅" in resultado_info.get("resultado", "") else ("v" if "🛑" in resultado_info.get("resultado", "") else "o")
                    ax.scatter([data_res_dt], [preco_res],
                               color=cor_r, s=120, marker=marker, zorder=7,
                               label=f"Resultado: {data_res_str}")
        except Exception:
            pass

    # Médias móveis (MM25 e MM55) como referência
    if len(df_plot) >= 25:
        mm25 = df_plot["Close"].rolling(25).mean()
        ax.plot(df_plot.index, mm25, color="#F38BA8", linewidth=0.8,
                linestyle="--", alpha=0.6, label="MM25", zorder=2)
    if len(df_plot) >= 55:
        mm55 = df_plot["Close"].rolling(55).mean()
        ax.plot(df_plot.index, mm55, color="#A6E3A1", linewidth=0.8,
                linestyle="--", alpha=0.6, label="MM55", zorder=2)

    # Formatação dos eixos
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=35, color="#CDD6F4", fontsize=9)
    plt.yticks(color="#CDD6F4", fontsize=9)

    # Grade
    ax.grid(axis="y", color="#313244", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", color="#313244", linewidth=0.3, alpha=0.4)
    ax.spines[:].set_color("#45475A")

    # Título e labels
    tk_clean = ticker.replace(".SA", "")
    resultado_str = resultado_info.get("resultado", "⏳")
    ret_real = resultado_info.get("retorno_realizado", None)
    ret_str  = f" | Retorno: {ret_real:+.1f}%" if ret_real is not None else ""
    ax.set_title(
        f"{tk_clean}  |  {resultado_str}{ret_str}",
        color="#CDD6F4", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_ylabel("Preço (R$)", color="#CDD6F4", fontsize=10)
    ax.set_xlabel("Data", color="#CDD6F4", fontsize=10)

    # Legenda
    legend = ax.legend(loc="upper left", framealpha=0.3,
                       facecolor="#313244", edgecolor="#45475A",
                       labelcolor="#CDD6F4", fontsize=8)

    plt.tight_layout(pad=1.5)

    # Converte para base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def gerar_html_graficos(backtest_list, dados_full, data_corte):
    """Gera bloco HTML com todos os gráficos em grid 1 coluna."""
    if not backtest_list:
        return "<p style='color:#888;font-family:sans-serif;'>Nenhum gráfico disponível.</p>"

    html = """
    <div style="font-family:'Segoe UI',sans-serif;background:#1E1E2E;
         padding:24px;border-radius:16px;margin-top:12px;">
      <h2 style="color:#CBA6F7;margin:0 0 20px 0;">📊 Gráficos de Preço por Ativo</h2>
    """

    for r in sorted(backtest_list, key=lambda x: x["score"], reverse=True):
        ticker     = r["ticker"]
        tk_clean   = ticker.replace(".SA", "")
        entrada    = r.get("preco", 0)
        alvo       = r.get("alvo", 0)
        stop_price = entrada * 0.90
        cor_card   = r.get("cor", "#45475A")

        img_b64 = plot_ativo_base64(
            ticker, dados_full, data_corte,
            entrada, alvo, stop_price, r
        )

        if img_b64 is None:
            continue

        resultado = r.get("resultado", "—")
        estrateg  = r.get("estrategia", "")
        score     = r.get("score", 0)
        ret_real  = r.get("retorno_realizado", None)
        ret_str   = f"{ret_real:+.1f}%" if ret_real is not None else "—"
        ret_cor   = "#00C851" if (ret_real or 0) > 0 else "#FF4444"

        html += f"""
        <div style="background:#313244;border-radius:14px;padding:16px;
             margin-bottom:20px;border-left:4px solid {cor_card};">
          <div style="display:flex;justify-content:space-between;align-items:center;
               flex-wrap:wrap;gap:8px;margin-bottom:12px;">
            <div>
              <span style="font-size:20px;font-weight:800;color:#CDD6F4;">{tk_clean}</span>
              <span style="margin-left:10px;color:#888;font-size:12px;">{estrateg}</span>
            </div>
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                Score: <b style="color:#CBA6F7;">{score}</b>
              </span>
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                {resultado}
              </span>
              <span style="background:#1E1E2E;padding:3px 10px;border-radius:7px;font-size:13px;">
                Retorno: <b style="color:{ret_cor};">{ret_str}</b>
              </span>
            </div>
          </div>
          <img src="data:image/png;base64,{img_b64}"
               style="width:100%;border-radius:8px;display:block;" />
        </div>"""

    html += "</div>"
    return html


# Gera e exibe os gráficos
if FAZER_BACKTEST and resultados_backtest and DADOS_FULL:
    print("📊 Gerando gráficos para cada ativo recomendado...")
    html_graficos = gerar_html_graficos(resultados_backtest, DADOS_FULL, DATA_CORTE)
    displayHTML(html_graficos)
    print("✅ Gráficos gerados!")
elif FAZER_BACKTEST and resultados_filtrados and not DADOS_FULL:
    print("⚠️ Dados completos não disponíveis para gerar gráficos.")
else:
    print("🔵 Gráficos de backtesting não aplicáveis (data atual).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏁 Fim do Notebook BNF Simulator
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📌 Resumo do que foi feito:
# MAGIC 1. ✅ Data de corte validada e dados históricos baixados da B3
# MAGIC 2. ✅ Indicadores calculados (MM9/21/25/55/200, Kairi, RSI, ATR, Bollinger, Volume Ratio)
# MAGIC 3. ✅ Regime de mercado detectado automaticamente (Bear / Neutro / Bull)
# MAGIC 4. ✅ 5 estratégias BNF aplicadas em todos os ativos
# MAGIC 5. ✅ Recomendações exibidas com explicação didática e score de confiança
# MAGIC 6. ✅ Backtesting automático (se data de corte no passado)
# MAGIC 7. ✅ Gráficos individuais com linha de corte, alvo e stop

# COMMAND ----------

# MAGIC %md
# MAGIC > ⚠️ **Lembre-se sempre dos 7 princípios de BNF:**
# MAGIC > 1. Nunca opere sem stop-loss
# MAGIC > 2. Liquidez é obrigatória
# MAGIC > 3. O mercado está sempre certo
# MAGIC > 4. Paciência é vantagem
# MAGIC > 5. Controle emocional
# MAGIC > 6. Diversifique as posições
# MAGIC > 7. O preço sempre reverte à média
# MAGIC =======
# MAGIC
# MAGIC >>>>>>> Stashed changes

# COMMAND ----------


