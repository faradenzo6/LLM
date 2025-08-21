# ai-gateway/Dockerfile
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1


# Базовые либы для WeasyPrint + шрифты
RUN apt-get update && apt-get install -y --no-install-recommends \
libpango-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi8 libxml2 libxslt1.1 \
fonts-dejavu-core \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Python deps сначала — для кеша слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Код приложения
COPY main.py /app/
COPY routes /app/routes
COPY static /app/static


# Каталоги под шаблоны и логи — будут монтироваться томами
RUN mkdir -p /app/doc-templates /var/log/ai-gateway \
&& adduser --disabled-password --gecos "" appuser \
&& chown -R appuser:appuser /app /var/log/ai-gateway


USER appuser
EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
CMD python - <<'PY' || exit 1
import json,urllib.request
try:
with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as r:
j=json.loads(r.read().decode('utf-8'))
assert j.get('status')=='ok'
except Exception:
raise SystemExit(1)
PY


# Uvicorn с указанием каталога приложения
CMD ["uvicorn","main:app","--app-dir","/app","--host","0.0.0.0","--port","8000"]
