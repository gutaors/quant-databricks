# 📋 Passo a Passo: Pipeline de Cotações no Databricks Free Edition

## Visão Geral

Este pipeline coleta cotações intraday de todos os seus tickers **a cada 30 minutos**
durante o pregão B3 (Seg–Sex, 10h–17h). A Free Edition do Databricks **suporta Jobs
com agendamento**, então podemos configurar tudo direto pela UI.

> **Limitações da Free Edition:**
> - Máximo de **5 tasks concorrentes** por conta
> - Apenas **serverless compute** (sem cluster customizado)
> - Cotas diárias/mensais de uso — se exceder, o workspace pode ser suspenso temporariamente

---

## Passo 1: Importar o Notebook

1. Acesse seu workspace Databricks
2. No menu lateral, clique em **Workspace** → sua pasta de usuário
3. Clique em **⋮** (três pontos) → **Import**
4. Selecione o arquivo `pipeline_cotacoes.py`
5. Confirme a importação

---

## Passo 2: Testar Manualmente

Antes de agendar, execute o notebook uma vez para garantir que tudo funciona:

1. Abra o notebook `pipeline_cotacoes`
2. Clique em **Run All** (▶️)
3. Verifique que:
   - Os tickers são encontrados (seção 1)
   - As cotações são baixadas (seção 2)
   - A tabela `cotacoes_intraday` é criada (seção 3)

> ⚠️ Se executar fora do pregão (antes das 10h, depois das 17h, ou fim de semana),
> o notebook vai parar na seção 0 com a mensagem "Fora do horário de pregão".
> Para testar, comente temporariamente o bloco de verificação.

---

## Passo 3: Criar o Job Agendado

### 3.1 — Abrir o painel de Jobs

1. No menu lateral esquerdo, clique em **Workflows** (ícone de fluxo/setas)
2. Clique no botão **Create Job** (canto superior direito)

### 3.2 — Configurar a Task

1. **Task name**: `pipeline_cotacoes_intraday`
2. **Type**: Notebook
3. **Source**: Workspace
4. **Path**: Navegue até o notebook `pipeline_cotacoes` na sua pasta
5. **Compute**: Selecione **Serverless** (única opção na Free Edition)

### 3.3 — Configurar o Agendamento (Schedule)

1. Na parte superior do Job, clique em **Add trigger**
2. Selecione **Scheduled**
3. Escolha o modo **Cron** (avançado)
4. Cole esta expressão cron:

```
0 */30 10-17 ? * MON-FRI
```

> **Tradução**: A cada 30 minutos, das 10h às 17h, de segunda a sexta.

5. **Timezone**: Selecione `America/Sao_Paulo` (BRT)
6. Clique em **Save**

> **Nota sobre a sintaxe cron do Databricks:**
> O Databricks usa Quartz Cron (6 campos, não 5), diferente do cron do Linux:
> ```
> ┌─ segundos (0-59)       → 0 (no segundo 0)
> │  ┌─ minutos (0-59)     → */30 (a cada 30 min)
> │  │  ┌─ horas (0-23)    → 10-17 (10h às 17h)
> │  │  │  ┌─ dia do mês   → ? (qualquer)
> │  │  │  │ ┌─ mês        → * (todos)
> │  │  │  │ │ ┌─ dia da semana → MON-FRI
> │  │  │  │ │ │
> 0 */30 10-17 ? * MON-FRI
> ```

### 3.4 — Nomear e Salvar o Job

1. No topo da página, clique no nome do job e renomeie para:
   **`Pipeline Cotações Intraday (30min)`**
2. Clique em **Create** ou **Save**

---

## Passo 4: Verificar que o Job está Ativo

1. Volte em **Workflows** no menu lateral
2. Você verá o job listado com o próximo horário de execução
3. O status deve estar como **Active**

---

## Passo 5: Monitorar Execuções

### Pela UI do Databricks:
1. Vá em **Workflows** → clique no seu job
2. A aba **Runs** mostra todas as execuções passadas
3. Cada run mostra: horário, duração, status (Success/Failed)
4. Clique em uma run para ver os logs detalhados

### Por SQL no Databricks:
```sql
-- Ver todas as coletas de hoje
SELECT * FROM workspace.default.cotacoes_intraday
WHERE Date = current_date()
ORDER BY Hora_Cotacao, Ticker;

-- Evolução de um ticker ao longo do dia
SELECT Hora_Cotacao, Close, Volume
FROM workspace.default.cotacoes_intraday
WHERE Ticker = 'PETR4.SA' AND Date = current_date()
ORDER BY Hora_Cotacao;

-- Resumo por coleta
SELECT Hora_Cotacao, COUNT(*) as Tickers, ROUND(AVG(Close), 2) as Media_Close
FROM workspace.default.cotacoes_intraday
WHERE Date = current_date()
GROUP BY Hora_Cotacao
ORDER BY Hora_Cotacao;

-- Variação intraday
SELECT Ticker,
       FIRST(Close) as Abertura,
       LAST(Close) as Ultimo,
       ROUND((LAST(Close) - FIRST(Close)) / FIRST(Close) * 100, 2) as Variacao_Pct
FROM workspace.default.cotacoes_intraday
WHERE Date = current_date()
GROUP BY Ticker
ORDER BY Variacao_Pct DESC;
```

---

## Passo 6 (Opcional): Alertas de Falha

1. Na configuração do Job, clique em **Edit**
2. Vá em **Notifications**
3. Adicione seu email para receber alertas quando o job **falhar**
4. Salve

---

## ⚠️ Limites da Free Edition

| Recurso | Limite |
|---------|--------|
| Tasks concorrentes | Máx **5** |
| Compute | Apenas **Serverless** |
| Cotas | Diárias/mensais (suspende se exceder) |
| Uso comercial | ❌ Proibido |

Com 15 coletas/dia × ~30s cada, o consumo é bem baixo e deve ficar dentro das cotas.

---

## 🔄 Alternativa: Execução Manual

Se preferir não usar o Job automático:

1. Abra o notebook `pipeline_cotacoes` no Databricks
2. Clique em **Run All** (▶️) manualmente
3. Repita a cada 30 minutos durante o pregão

Cada execução faz append — os dados se acumulam automaticamente.
