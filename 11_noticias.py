# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 📰 Notícias do Mercado
# MAGIC Busca notícias do Google News para ações do portfólio.
# MAGIC
# MAGIC Equivalente a `pages/notícias.py` do Streamlit.

# COMMAND ----------

# MAGIC %pip install beautifulsoup4 --quiet

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd

# COMMAND ----------

# ── Widgets ───────────────────────────────────────────────────────────────────

tickers_list = listar_tickers_disponiveis(spark)
if tickers_list:
    dbutils.widgets.dropdown("ticker", tickers_list[0], tickers_list, "Selecione o Ticker")
else:
    dbutils.widgets.text("ticker", "PETR4.SA", "Digite o Ticker")

dbutils.widgets.dropdown("periodo", "ultimas", ["ultimas", "data_especifica"], "Período")
dbutils.widgets.text("data_busca", datetime.now().strftime("%Y-%m-%d"), "Data Específica (AAAA-MM-DD)")

# COMMAND ----------

# ── Funções ───────────────────────────────────────────────────────────────────

def get_company_name(ticker):
    """Retorna o nome da empresa baseado no ticker."""
    nomes = {
        "PETR4.SA": "Petrobras", "VALE3.SA": "Vale", "ITUB4.SA": "Itaú Unibanco",
        "BBDC4.SA": "Bradesco", "ABEV3.SA": "Ambev", "B3SA3.SA": "B3",
        "BBAS3.SA": "Banco do Brasil", "WEGE3.SA": "WEG", "RENT3.SA": "Localiza",
        "SUZB3.SA": "Suzano", "MGLU3.SA": "Magazine Luiza", "CSUD3.SA": "CSU Digital",
        "EMBR3.SA": "Embraer"
    }
    base = ticker.replace(".SA", "").replace(".sa", "")
    return nomes.get(ticker, base)

def clean_html(text):
    """Remove tags HTML do texto."""
    if text:
        soup = BeautifulSoup(text, "html.parser")
        clean = soup.get_text()
        return clean[:200] + "..." if len(clean) > 200 else clean
    return ""

def search_google_news(company_name, ticker, selected_date=None, last_30_days=False):
    """Busca notícias no Google News via RSS."""
    base_name = ticker.replace(".SA", "").replace(".sa", "")
    query = f"{company_name} OR {base_name} ação bolsa"
    query_encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=10)
        content = response.read()
        root = ET.fromstring(content)

        articles = []
        date_limit = datetime.now() - timedelta(days=30)

        for item in root.findall(".//item"):
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")
            description_elem = item.find("description")
            source_elem = item.find("source")

            if title_elem is None or link_elem is None:
                continue

            title = title_elem.text
            link = link_elem.text
            pub_date = pub_date_elem.text if pub_date_elem is not None else ""
            description = description_elem.text if description_elem is not None else ""
            source = source_elem.text if source_elem is not None else "Google News"

            try:
                article_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            except:
                try:
                    article_date = datetime.strptime(pub_date.split(" GMT")[0], "%a, %d %b %Y %H:%M:%S")
                except:
                    article_date = None

            include = False
            if last_30_days:
                include = article_date is None or article_date >= date_limit
            elif selected_date:
                include = article_date is None or (article_date and article_date.date() == selected_date)
            else:
                include = True

            if include:
                articles.append({
                    "Titulo": title,
                    "Fonte": source,
                    "Data": pub_date,
                    "Descricao": clean_html(description),
                    "Link": link
                })

        return articles

    except Exception as e:
        print(f"Erro ao buscar notícias: {e}")
        return []

# COMMAND ----------

# ── Buscar Notícias ───────────────────────────────────────────────────────────

ticker = dbutils.widgets.get("ticker")
periodo = dbutils.widgets.get("periodo")
data_busca = dbutils.widgets.get("data_busca")

company_name = get_company_name(ticker)

displayHTML(f"<h2>📰 Notícias — {company_name} ({ticker})</h2>")

if periodo == "ultimas":
    articles = search_google_news(company_name, ticker, last_30_days=True)
    displayHTML(f"<p>Buscando notícias dos últimos 30 dias...</p>")
else:
    sel_date = pd.to_datetime(data_busca).date()
    articles = search_google_news(company_name, ticker, selected_date=sel_date)
    displayHTML(f"<p>Buscando notícias para {formatar_data(sel_date)}...</p>")

# COMMAND ----------

# ── Exibir Resultados ─────────────────────────────────────────────────────────

if articles:
    displayHTML(html_alert(f"Encontradas <b>{len(articles)}</b> notícia(s)", "success"))

    # Exibir como DataFrame
    df_noticias = pd.DataFrame(articles)
    display(spark.createDataFrame(df_noticias))

    # Exibir em formato HTML rico
    html_articles = "<hr/>"
    for idx, art in enumerate(articles, 1):
        html_articles += f"""
        <div style='background:#1e1e1e; padding:16px; border-radius:8px; margin:12px 0;'>
            <p style='color:#aaa; margin:0;'><b>{art['Fonte']}</b></p>
            <h3 style='color:white; margin:4px 0;'>Notícia {idx}</h3>
            <p style='color:#e0e0e0; margin:4px 0;'><b>{art['Titulo']}</b></p>
            <p style='color:#bbb; margin:4px 0;'>{art['Descricao']}</p>
            <p style='color:#888; margin:4px 0;'>📅 {art['Data']}</p>
            <a href='{art['Link']}' target='_blank' style='color:#4fc3f7;'>🔗 Ler notícia completa</a>
        </div>
        """
    displayHTML(html_articles)
else:
    displayHTML(html_alert("Nenhuma notícia encontrada. Tente outro ticker ou período.", "warning"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ℹ️ Sobre
# MAGIC **Fonte:** Google News (Brasil)
# MAGIC
# MAGIC As notícias são coletadas do Google News com filtro para o Brasil e idioma português brasileiro.
