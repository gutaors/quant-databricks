# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 🏛️ Órgãos Unificados
# MAGIC Visualização dos órgãos e entidades da APF direta e indireta,
# MAGIC unificados e padronizados pelos códigos das diversas fontes.
# MAGIC
# MAGIC Equivalente a `pages/orgaos_unificados_static.py` do Streamlit.

# COMMAND ----------

# MAGIC %run ./utils_quant

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar Dados dos Órgãos

# COMMAND ----------

# Carregar dados da tabela de órgãos unificados
try:
    df_org = spark.table(TABELA_ORGAOS).toPandas()
    print(f"Dados carregados: {len(df_org)} registros")
except Exception as e:
    displayHTML(html_alert(f"Tabela <b>{TABELA_ORGAOS}</b> não encontrada. Verifique se os dados foram importados.", "error"))
    dbutils.notebook.exit("Sem dados")

# Normalizar nomes de colunas
df_org.columns = df_org.columns.str.lower()

# Renomear colunas se necessário (mapeia dos nomes originais do CSV)
rename_map = {
    "org_super_padr_nome": "super_padr_nome",
    "org_padr_nome": "padr_nome",
    "org_padr_sigla": "padr_sigla",
    "org_padr_codigo": "padr_codigo",
    "orgao_unificado_id": "unif_id",
    "orgao_unificado_fonte": "fonte",
    "orgao_unificado_id_origem": "id_origem",
    "orgao_unificado_nome": "nome",
    "orgao_unificado_sigla": "sigla",
    "orgao_unificado_situacao": "situacao",
    "orgao_unificado_cod_siorg_origem": "cod_siorg",
    "org_padr_id": "padr_id",
    "orgao_unificado_dt_atualizacao": "atualizacao"
}
# Only rename columns that exist
existing_renames = {k: v for k, v in rename_map.items() if k in df_org.columns}
if existing_renames:
    df_org.rename(existing_renames, axis=1, inplace=True)

# Converter tipos
for col in ["padr_codigo", "unif_id", "id_origem", "cod_siorg", "padr_id"]:
    if col in df_org.columns:
        df_org[col] = df_org[col].astype("Int64").astype(str)

displayHTML(f"<h2>🏛️ Órgãos Unificados</h2><p>{len(df_org)} registros carregados</p>")

# COMMAND ----------

# ── Widgets de Filtro ─────────────────────────────────────────────────────────

TODOS = "<Todos>"

# Filtro 1: Órgão Superior
if "org_super_busca" in df_org.columns:
    sup_col = "org_super_busca"
elif "super_padr_nome" in df_org.columns:
    sup_col = "super_padr_nome"
else:
    sup_col = df_org.columns[0]

sup_list = sorted(df_org[sup_col].dropna().unique().tolist())
sup_list.insert(0, TODOS)
dbutils.widgets.dropdown("orgao_superior", TODOS, sup_list, "Órgão Superior")

# COMMAND ----------

# ── Aplicar Filtros ───────────────────────────────────────────────────────────

filtro_sup = dbutils.widgets.get("orgao_superior")

df_filtrado = df_org.copy()
if filtro_sup != TODOS:
    df_filtrado = df_filtrado[df_filtrado[sup_col] == filtro_sup]

# Métricas
qtde_sup = df_filtrado["super_padr_nome"].nunique() if "super_padr_nome" in df_filtrado.columns else 0
qtde_org = df_filtrado["padr_nome"].nunique() if "padr_nome" in df_filtrado.columns else 0

displayHTML(f"""
<div style='display:flex; gap:8px; margin:16px 0;'>
    {html_metric("Órgãos Superiores", str(qtde_sup))}
    {html_metric("Órgãos/Entidades", str(qtde_org))}
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Órgão Padronizado

# COMMAND ----------

if filtro_sup != TODOS:
    cols_padr = [c for c in ["padr_id", "super_padr_nome", "padr_nome", "padr_sigla", "padr_codigo"] if c in df_filtrado.columns]
    if cols_padr:
        df_padr = df_filtrado.groupby(cols_padr, as_index=False).size()
        displayHTML("<h3>Órgãos Padronizados</h3>")
        display(spark.createDataFrame(df_padr[cols_padr]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Órgão Unificado — Por Fonte

# COMMAND ----------

if filtro_sup != TODOS and "fonte" in df_filtrado.columns:
    cols_unif = [c for c in ["unif_id", "id_origem", "nome", "sigla", "situacao", "cod_siorg", "padr_id", "atualizacao"] if c in df_filtrado.columns]

    fontes = df_filtrado["fonte"].dropna().unique().tolist()

    for fonte in sorted(fontes):
        df_fonte = df_filtrado[df_filtrado["fonte"] == fonte]
        if not df_fonte.empty:
            displayHTML(f"<h3>Fonte: {fonte}</h3><p>{len(df_fonte)} registros</p>")
            display(spark.createDataFrame(df_fonte[cols_unif] if cols_unif else df_fonte))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datamart SIORG (Lookup)

# COMMAND ----------

if filtro_sup != TODOS and "fonte" in df_filtrado.columns:
    df_siorg_filtrado = df_filtrado[df_filtrado["fonte"] == "SIORG"]
    if not df_siorg_filtrado.empty and "cod_siorg" in df_siorg_filtrado.columns:
        uo_list = df_siorg_filtrado["cod_siorg"].dropna().unique().tolist()

        if uo_list:
            try:
                df_dm = spark.table(TABELA_SIORG).toPandas()
                df_dm.columns = df_dm.columns.str.lower()

                # Renomear colunas
                dm_renames = {
                    "id_unidade_organizacional": "id",
                    "co_unidade_organizacional": "cod_siorg",
                    "no_unidade_organizacional": "nome",
                    "sg_unidade_organizacional": "sigla",
                    "in_tipo_unidade_organizacional": "tipo",
                    "sn_ativo": "ativo",
                    "dt_criacao": "criacao",
                    "dt_alteracao": "alteracao",
                    "no_categoria": "categoria",
                    "no_natureza_juridica": "natureza",
                    "no_subnatureza_juridica": "subnatureza",
                }
                existing_dm = {k: v for k, v in dm_renames.items() if k in df_dm.columns}
                if existing_dm:
                    df_dm.rename(existing_dm, axis=1, inplace=True)

                if "cod_siorg" in df_dm.columns:
                    df_dm["cod_siorg"] = df_dm["cod_siorg"].astype("Int64").astype(str)
                    df_dm_filtered = df_dm[df_dm["cod_siorg"].isin(uo_list)]

                    if not df_dm_filtered.empty:
                        displayHTML("<h3>Datamart SIORG</h3>")
                        display(spark.createDataFrame(df_dm_filtered))
                    else:
                        displayHTML(html_alert("Sem dados no Datamart SIORG para os códigos encontrados.", "info"))
            except Exception as e:
                displayHTML(html_alert(f"Tabela {TABELA_SIORG} não encontrada: {e}", "warning"))
