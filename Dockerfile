# Playwright needs Chromium + system libs. The official Playwright Python image
# ships them preinstalled.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

WORKDIR /app

# Install Python deps first so the layer caches across code changes. Installing
# Chromium again makes sure the browser matches the Playwright version pip chose.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install chromium

COPY . .

# Serve on all interfaces inside the container; map the port in compose. The
# container has no display, so never try to pop a login window: answer with a
# 503 telling the caller to sign in on the host instead.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    SERVER_INTERACTIVE_LOGIN=0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "app.py"]
