#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# agendar_pipeline.sh
# Executa o notebook pipeline_cotacoes no Databricks via REST API.
# Use com crontab ou Task Scheduler para agendamento automático.
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIGURAÇÃO (preencha com seus dados) ────────────────────────────────────
DATABRICKS_HOST="https://community.cloud.databricks.com"  # ou seu workspace URL
DATABRICKS_TOKEN="dapi_SEU_TOKEN_AQUI"                     # Personal Access Token
NOTEBOOK_PATH="/Users/SEU_EMAIL/quant-databricks/pipeline_cotacoes"  # caminho no workspace
CLUSTER_ID="SEU_CLUSTER_ID"                                 # ID do cluster

# ── VALIDAÇÕES ────────────────────────────────────────────────────────────────
if [[ "$DATABRICKS_TOKEN" == *"SEU_TOKEN"* ]]; then
    echo "❌ ERRO: Configure o DATABRICKS_TOKEN neste script."
    echo "   Siga o README_pipeline.md para gerar um token."
    exit 1
fi

if [[ "$CLUSTER_ID" == *"SEU_CLUSTER"* ]]; then
    echo "❌ ERRO: Configure o CLUSTER_ID neste script."
    exit 1
fi

# ── VERIFICAR SE CURL ESTÁ DISPONÍVEL ─────────────────────────────────────────
if ! command -v curl &> /dev/null; then
    echo "❌ curl não encontrado. Instale: sudo apt install curl"
    exit 1
fi

# ── HORA ATUAL ────────────────────────────────────────────────────────────────
HORA=$(TZ="America/Sao_Paulo" date "+%Y-%m-%d %H:%M:%S")
echo "🕐 [$HORA] Iniciando pipeline de cotações..."

# ── VERIFICAR SE O CLUSTER ESTÁ ATIVO ─────────────────────────────────────────
echo "   Verificando estado do cluster..."
CLUSTER_STATE=$(curl -s -X GET \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    "$DATABRICKS_HOST/api/2.0/clusters/get?cluster_id=$CLUSTER_ID" \
    | python3 -c "import sys, json; print(json.load(sys.stdin).get('state', 'UNKNOWN'))" 2>/dev/null)

echo "   Estado do cluster: $CLUSTER_STATE"

if [[ "$CLUSTER_STATE" != "RUNNING" ]]; then
    echo "   ⏳ Iniciando cluster..."
    curl -s -X POST \
        -H "Authorization: Bearer $DATABRICKS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"cluster_id\": \"$CLUSTER_ID\"}" \
        "$DATABRICKS_HOST/api/2.0/clusters/start" > /dev/null

    # Aguardar cluster iniciar (máx 10 minutos)
    for i in $(seq 1 60); do
        sleep 10
        STATE=$(curl -s -X GET \
            -H "Authorization: Bearer $DATABRICKS_TOKEN" \
            "$DATABRICKS_HOST/api/2.0/clusters/get?cluster_id=$CLUSTER_ID" \
            | python3 -c "import sys, json; print(json.load(sys.stdin).get('state', 'UNKNOWN'))" 2>/dev/null)
        echo "   [$i/60] Estado: $STATE"
        if [[ "$STATE" == "RUNNING" ]]; then
            echo "   ✅ Cluster ativo!"
            break
        fi
    done
fi

# ── EXECUTAR NOTEBOOK ─────────────────────────────────────────────────────────
echo "   🚀 Executando notebook: $NOTEBOOK_PATH"

# Criar contexto de execução
CONTEXT_ID=$(curl -s -X POST \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"language\": \"python\", \"clusterId\": \"$CLUSTER_ID\"}" \
    "$DATABRICKS_HOST/api/1.2/contexts/create" \
    | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null)

if [[ -z "$CONTEXT_ID" ]]; then
    echo "❌ Falha ao criar contexto de execução."
    exit 1
fi

echo "   Context ID: $CONTEXT_ID"

# Executar comando para rodar o notebook
COMMAND_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
        \"language\": \"python\",
        \"clusterId\": \"$CLUSTER_ID\",
        \"contextId\": \"$CONTEXT_ID\",
        \"command\": \"dbutils.notebook.run('$NOTEBOOK_PATH', 600)\"
    }" \
    "$DATABRICKS_HOST/api/1.2/commands/execute")

COMMAND_ID=$(echo "$COMMAND_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null)
echo "   Command ID: $COMMAND_ID"

# Aguardar conclusão (máx 10 minutos)
echo "   ⏳ Aguardando conclusão..."
for i in $(seq 1 60); do
    sleep 10
    STATUS_RESPONSE=$(curl -s -X GET \
        -H "Authorization: Bearer $DATABRICKS_TOKEN" \
        "$DATABRICKS_HOST/api/1.2/commands/status?clusterId=$CLUSTER_ID&contextId=$CONTEXT_ID&commandId=$COMMAND_ID")

    STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'Unknown'))" 2>/dev/null)

    if [[ "$STATUS" == "Finished" ]]; then
        RESULT=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('results', {})
print(f\"Type: {r.get('resultType', '?')} | Data: {r.get('data', 'N/A')}\")
" 2>/dev/null)
        echo "   ✅ Concluído! $RESULT"
        break
    elif [[ "$STATUS" == "Error" || "$STATUS" == "Cancelled" ]]; then
        echo "   ❌ Falha na execução: $STATUS"
        echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('results', {})
print(f\"   Causa: {r.get('cause', 'desconhecida')}\")
" 2>/dev/null
        break
    else
        echo "   [$i/60] Status: $STATUS"
    fi
done

# Destruir contexto
curl -s -X POST \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"contextId\": \"$CONTEXT_ID\", \"clusterId\": \"$CLUSTER_ID\"}" \
    "$DATABRICKS_HOST/api/1.2/contexts/destroy" > /dev/null

HORA_FIM=$(TZ="America/Sao_Paulo" date "+%Y-%m-%d %H:%M:%S")
echo "🏁 [$HORA_FIM] Pipeline finalizado."
