import os, re, requests, collections
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# наши рабочие роуты
from routes import doc_edit

VLLM_URL   = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
VLLM_KEY   = os.getenv("VLLM_API_KEY", "sk-local-jira-test-123")
MODEL      = os.getenv("VLLM_MODEL", "Qwen/Qwen2-7B-Instruct")
TEI_URL    = os.getenv("TEI_URL", "http://host.docker.internal:8080/embed")
QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")

app = FastAPI(title="AI Gateway")

@app.get("/")
def root(): return JSONResponse({"status":"running"})
@app.get("/health")
def health(): return {"status":"ok"}

# ====== Schemas ======
class SummarizeIn(BaseModel): text: str
class SummarizeOut(BaseModel): summary: str
class ChecklistIn(BaseModel): text: str
class ChecklistOut(BaseModel): steps: list[str]
class CommentIn(BaseModel): text: str
class CommentOut(BaseModel): comment: str

class KBAskIn(BaseModel):
    question: str
    collection: str = "jira-embeddings"
    limit: int = 5

class KBHit(BaseModel):
    id: str | int | None = None
    score: float | None = None
    payload: dict | None = None

class KBAskOut(BaseModel):
    answer: str
    hits: list[KBHit]

# ====== Helpers ======
def _llm(messages: list[dict], max_tokens=256, temperature=0.2) -> str:
    r = requests.post(
        VLLM_URL,
        headers={"Authorization": f"Bearer {VLLM_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=120,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM error {r.status_code}: {r.text[:1000]}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def _parse_steps(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps = []
    for l in lines:
        if re.match(r"^(\d+[\).\:\-]\s+|\-\s+)", l):
            l = re.sub(r"^(\d+[\).\:\-]\s+|\-\s+)", "", l).strip()
            steps.append(l)
    if not steps:
        parts = re.split(r"[.!?]\s+", text.strip())
        steps = [p.strip() for p in parts if p.strip()]
    steps = steps[:6]
    return [s[:240] for s in steps if s]

def _embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(TEI_URL, json={"inputs": texts}, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TEI error {r.status_code}: {r.text[:500]}")
    return r.json()

def _qdrant_search(collection: str, vector: list[float], limit: int = 5) -> dict:
    r = requests.post(
        f"{QDRANT_URL}/collections/{collection}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True},
        timeout=60,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Qdrant error {r.status_code}: {r.text[:500]}")
    return r.json()

# ====== Endpoints ======
@app.post("/summarize", response_model=SummarizeOut)
def summarize(body: SummarizeIn):
    prompt_sys = (
        "Ты помощник Jira. Проанализируй все данные задачи и верни строго 4–5 предложений: "
        "1) краткая суть, 2) что требуется сделать (конкретно), 3) ключевые риски/блокеры если есть. "
        "Пиши по-русски, без лишней воды."
    )
    content = _llm(
        [{"role": "system", "content": prompt_sys},
         {"role": "user", "content": body.text}],
        max_tokens=256, temperature=0.2
    )
    return {"summary": content}

@app.post("/checklist", response_model=ChecklistOut)
def checklist(body: ChecklistIn):
    prompt_sys = (
        "Ты помощник Jira. Сгенерируй короткий чек-лист для исполнителя по задаче. "
        "Формат — нумерованный список, 1 пункт = 1 действие, максимум 6 пунктов. "
        "Каждый пункт в повелительном наклонении и максимально конкретен. Без воды."
    )
    content = _llm(
        [{"role": "system", "content": prompt_sys},
         {"role": "user", "content": body.text}],
        max_tokens=256, temperature=0.2
    )
    return {"steps": _parse_steps(content)}

@app.post("/comment", response_model=CommentOut)
def comment(body: CommentIn):
    prompt_sys = (
        "Ты помощник Jira. На основе описания, статуса и истории комментариев подготовь ОДИН готовый комментарий от исполнителя. "
        "Если задача выполнена — кратко подтвердить выполнение и перечислить сделанные действия. "
        "Если не хватает данных — вежливо запросить ЗАПОЛНИТЬ конкретные поля/данные (через «; »). "
        "Если нужен следующий шаг — предложи его конкретно. "
        "Стиль: деловой, без приветствий/подписей, максимум 4 предложения."
    )
    content = _llm(
        [{"role": "system", "content": prompt_sys},
         {"role": "user", "content": body.text}],
        max_tokens=180, temperature=0.2
    )
    return {"comment": content[:800]}

@app.post("/kb_ask", response_model=KBAskOut)
def kb_ask(body: KBAskIn):
    try:
        vec = _embed([body.question])[0]
        sr = _qdrant_search(body.collection, vec, body.limit)
        hits = sr.get("result", []) or []
        ctx_lines = []
        approver_counter = collections.Counter()
        for h in hits:
            p = (h.get("payload") or {})
            issue = p.get("issue_key") or p.get("key") or ""
            itype = p.get("issue_type") or p.get("type") or ""
            txt = p.get("text") or ""
            approvers = p.get("approvers") or []
            if isinstance(approvers, list):
                approver_counter.update([a for a in approvers if isinstance(a, str)])
            ctx_lines.append(f"- [{itype}] {issue}: {txt[:500]}")
            if approvers:
                ctx_lines.append(f"  APPROVERS: {', '.join([str(a) for a in approvers][:10])}")
        context = "\n".join(ctx_lines) if ctx_lines else "(нет контекста)"

        prompt_sys = (
            "Ты справочный ассистент Jira. Используй только предоставленные примеры прошлых задач, чтобы ответить кратко и по делу. "
            "Если вопрос про согласование (кого добавлять), вычисли по контексту частых согласующих (поле APPROVERS) и выведи список вида: Имя — N раз. "
            "Если данных недостаточно — скажи, каких данных не хватает. Не выдумывай."
        )
        user_msg = f"Вопрос пользователя: {body.question}\n\nПримеры:\n{context}\n\nОтвет (не более 4 предложений):"
        answer = _llm(
            [{"role": "system", "content": prompt_sys},
             {"role": "user", "content": user_msg}],
            max_tokens=220, temperature=0.1
        )

        q_lower = body.question.lower()
        if any(w in q_lower for w in ["согласован", "согласование", "кого добав", "утвержден"]):
            top = approver_counter.most_common(5)
            if top:
                extra = "; ".join([f"{name} — {cnt} раз" for name, cnt in top])
                answer = f"{answer}\nЧаще всего добавляли: {extra}."

        kb_hits = []
        for h in hits:
            kb_hits.append({
                "id": h.get("id"),
                "score": h.get("score"),
                "payload": h.get("payload"),
            })

        return {"answer": answer.strip(), "hits": kb_hits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"kb_ask error: {e}")


# ====== статика /doc/ui ======
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
try:
    app.mount("/static", StaticFiles(directory="/app/static"), name="static")
except Exception as e:
    print("static mount failed:", e)

@app.get("/doc/ui")
def _doc_ui():
    return FileResponse("/app/static/doc-ui/index.html")

# ====== подключаем doc_edit ======
app.include_router(doc_edit.router)

from routes import jira_attach
app.include_router(jira_attach.router)
