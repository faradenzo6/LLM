# AI Stack: AI-gateway + vLLM + Qdrant + TEI

Инфраструктура для интеграции Jira ↔ AI-gateway ↔ LLM (vLLM), RAG через Qdrant и эмбеддинги через TEI. Есть UI для документов и API для генерации/редактирования.

## Архитектура

Jira Plugin → HTTPS → **Nginx (TLS, reverse proxy)** → **AI-gateway (FastAPI)**
- Генеративка → **vLLM** (OpenAI-совместимый API)
- Эмбеддинги → **TEI** (`/embed`)
- Векторное хранилище → **Qdrant**
- Документный UI → `/static/doc-ui/index.html`
- Аудит документных операций → `/var/log/ai-gateway/audit.jsonl`

Все сервисы в Docker-сети `ai-net`, ходят по DNS-имёнам: `vllm`, `qdrant`, `tei-embeddings`.

## Быстрый старт

**Предусловия:** Docker, Docker Compose, заполненный `.env` (на основе `.env.example`).

```bash
git clone <этот репозиторий>
cd <repo>
cp .env.example .env   # заполнить значения
docker compose up -d --build
Проверки
bash
Копировать
Редактировать
# AI-gateway
curl -s http://127.0.0.1:9000/health
curl -s -H "Content-Type: application/json" -H "X-API-Key: <AI_GATEWAY_API_KEY>" \
  -d '{"text":"проверка"}' http://127.0.0.1:9000/summarize

# vLLM
curl -s -H "Authorization: Bearer <VLLM_API_KEY>" http://127.0.0.1:8000/v1/models

# Qdrant
curl -s http://127.0.0.1:6333/collections | jq

# TEI
curl -s -H "Content-Type: application/json" -d '{"inputs":["test"]}' http://127.0.0.1:8080/embed | head

# UI
curl -I http://127.0.0.1:9000/static/doc-ui/index.html
Порты по умолчанию
ai-gateway: 9000→8000

vLLM: 8000→8000

qdrant: 6333-6334→6333-6334

tei-embeddings: 8080→80

Переменные окружения (пример)
makefile
Копировать
Редактировать
AI_GATEWAY_API_KEY=
VLLM_API_KEY=sk-local-jira-test-123
VLLM_MODEL=Qwen/Qwen2-7B-Instruct
AI_CORS_ORIGINS=https://itsm.uztelecom.uz,http://185.100.54.73
JIRA_BASE_URL=
JIRA_USER=
JIRA_PASS=
Реальный .env в репозиторий не коммитить.

Томá/персистентность
Qdrant: /qdrant/storage

Логи AI-gateway: /var/log/ai-gateway

Шаблоны: /app/doc-templates

Основные эндпоинты AI-gateway
GET /health

POST /summarize {text}

POST /checklist {text}

POST /comment {issue_key,tone,goal,hints,text}

POST /kb_ask {question,collection,top_k,answer_lang}
Документы/UI: GET /static/doc-ui/index.html, GET /doc/templates,
POST /doc/preview, POST /doc/generate, POST /doc/edit,
POST /doc/ai-ops, POST /doc/ai-edit.

CI/CD (GHCR)
Есть GitHub Actions workflow, который собирает образ ai-gateway и публикует в GHCR:
ghcr.io/<owner>/ai-gateway:<tag>.
Как включить — см. ниже «GHCR/Actions — как включить».

Обновление / откаты
bash
Копировать
Редактировать
# тянуть latest из GHCR
docker compose pull && docker compose up -d

# откат на конкретный тег/sha:
# в docker-compose.yml укажи image: ghcr.io/<owner>/ai-gateway:<tag>
docker compose up -d
GHCR/Actions — как включить
Разрешить workflow пушить пакеты
Repo → Settings → Actions → General → Workflow permissions: Read and write.

Workflow — .github/workflows/build.yml:

логин в ghcr.io через встроенный ${{ secrets.GITHUB_TOKEN }}

сборка из ./ai-gateway/Dockerfile

пуш тегов latest и ${{ github.sha }}

Где смотреть образ
GitHub → ваш профиль/организация → Packages → ai-gateway.

Как использовать в compose

yaml
Копировать
Редактировать
services:
  ai-gateway:
    image: ghcr.io/<owner>/ai-gateway:latest
    # ...
Приватный пакет
На сервере для docker pull сделай:

bash
Копировать
Редактировать
echo "<PAT с правами read:packages>" | docker login ghcr.io -u <github_user> --password-stdin
Или сделай пакет публичным в Package settings.

