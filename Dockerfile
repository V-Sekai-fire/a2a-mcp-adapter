FROM python:3.11-slim
WORKDIR /app
RUN useradd --system --no-create-home --uid 10001 app && mkdir /data && chown app /data
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
USER app
ENV A2A_MCP_DATA_DIR=/data
# Containers must listen on all interfaces; this requires A2A_MCP_AUTH_TOKEN for HTTP transports.
ENV MCP_HOST=0.0.0.0 A2A_HOST=0.0.0.0
EXPOSE 8000
ENTRYPOINT ["python", "a2a_mcp_server.py"]
