# ai-gateway/Dockerfile
FROM python:3.11.9-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8


# Базовые либы для WeasyPrint + шрифты
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi8 libxml2 libxslt1.1 fonts-dejavu-core \
    libpangoft2-1.0-0 libharfbuzz0b libjpeg62-turbo libpng16-16 tzdata shared-mime-info \
  && rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Python deps (любой из файлов)
COPY requirements* ./
RUN if [ -f requirements.lock ]; then \
      python -m pip install --no-cache-dir -r requirements.lock; \
    elif [ -f requirements.txt ]; then \
      python -m pip install --no-cache-dir -r requirements.txt; \
    else \
      echo "No requirements.lock or requirements.txt found" >&2; exit 1; \
    fi


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
  CMD ["python","-c","import urllib.request,sys; sys.exit(0 if urllib.request.urlopen(\"http://127.0.0.1:8000/health\", timeout=3).status==200 else 1)"]


# Uvicorn с указанием каталога приложения
CMD ["uvicorn","main:app","--app-dir","/app","--host","0.0.0.0","--port","8000","--proxy-headers","--forwarded-allow-ips","*"]
