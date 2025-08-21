from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Query
from starlette.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, Optional
import base64, os, json

router = APIRouter(prefix="/jira", tags=["jira"])
AUDIT_PATH = "/var/log/ai-gateway/audit.jsonl"

def _load_env_file(path: str) -> None:
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith("#") or "=" not in s: continue
                k,v=s.split("=",1); k=k.strip(); v=v.strip()
                if k and (k not in os.environ): os.environ[k]=v
    except Exception: pass

for p in ("/etc/ai-gateway.env","/app/.env"): _load_env_file(p)

def _audit_write(rec: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        with open(AUDIT_PATH,"a",encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    except Exception: pass

def _env_or(v: Optional[str], key: str) -> Optional[str]: return v or os.getenv(key)

@router.post("/attach")
async def attach(
    request: Request,
    issueKey: str = Query(...),
    file: UploadFile = File(None),
    file_base64: str = Form(None),
    filename: str = Form(None),
    content_type: str = Form(None),
    jira_url: str = Form(None),
    jira_user: str = Form(None),
    jira_pass: str = Form(None),
):
    ts = datetime.utcnow().isoformat(timespec="seconds")+"Z"
    ip = getattr(request.client,"host",None)
    x_user = request.headers.get("x-user") or request.headers.get("x-auth-user")

    jb = _env_or(jira_url, "JIRA_BASE_URL"); ju = _env_or(jira_user,"JIRA_USER"); jp = _env_or(jira_pass,"JIRA_PASS")
    if not jb or not ju or not jp:
        _audit_write({"ts":ts,"endpoint":"/jira/attach","ok":False,"error":"Missing Jira creds","ip":ip,"user":x_user,"issueKey":issueKey})
        raise HTTPException(status_code=400, detail="Missing Jira credentials")

    if file is not None:
        data = await file.read()
        if not data: raise HTTPException(status_code=400, detail="Empty uploaded file")
        fname = file.filename or filename or "document.bin"
        ctype = content_type or (file.content_type or "application/octet-stream")
    elif file_base64:
        try: data = base64.b64decode(file_base64)
        except Exception: raise HTTPException(status_code=400, detail="Invalid base64")
        fname = filename or "document.bin"; ctype = content_type or "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail="Provide either file or file_base64")

    url = f"{jb.rstrip('/')}/rest/api/2/issue/{issueKey}/attachments"
    headers = {"X-Atlassian-Token":"no-check"}
    try:
        try:
            import httpx
            files = {"file": (fname, data, ctype)}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, files=files, auth=(ju, jp))
                status, text = resp.status_code, resp.text
        except Exception:
            import requests
            files = {"file": (fname, data, ctype)}
            r = requests.post(url, headers=headers, files=files, auth=(ju, jp), timeout=60)
            status, text = r.status_code, r.text
    except Exception as e:
        _audit_write({"ts":ts,"endpoint":"/jira/attach","ok":False,"error":str(e),"ip":ip,"user":x_user,"issueKey":issueKey,"filename":fname})
        raise HTTPException(status_code=502, detail=f"Jira attach failed: {e}")

    ok = 200 <= status < 300
    _audit_write({"ts":ts,"endpoint":"/jira/attach","ok":ok,"ip":ip,"user":x_user,
                  "issueKey":issueKey,"filename":fname,"bytes":len(data),"status":status})
    return JSONResponse({"ok":ok,"status":status,"issueKey":issueKey,"filename":fname,"jira_response":text})
