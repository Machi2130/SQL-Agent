FROM python:3.12-slim

# ODBC Driver 17 for SQL Server (needed by pyodbc)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 unixodbc \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUTF8=1
# Learned schema knowledge lives here — mount a volume to keep it across restarts.
VOLUME ["/app/.chroma_data"]
EXPOSE 8000
CMD ["python", "mcp_server.py", "--http", "0.0.0.0:8000"]
