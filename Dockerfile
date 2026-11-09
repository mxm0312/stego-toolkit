# Лёгкий образ: LSB + веб-интерфейс, без PyTorch и CUDA.
FROM python:3.12-slim-bookworm

# ffmpeg собирает браузерный MP4 из основной и скрытой дорожек.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY stego ./stego

# --frozen: ставим строго версии из uv.lock (без пере-резолва) — воспроизводимая сборка.
# Пакет в /app/.venv, venv в PATH — stego/python/streamlit доступны напрямую.
RUN uv sync --extra ui --frozen --no-cache --no-editable --no-dev
ENV PATH="/app/.venv/bin:$PATH" \
    STEGO_ENGINES=lsb

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
