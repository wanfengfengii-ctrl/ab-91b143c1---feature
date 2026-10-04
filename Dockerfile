FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# 先复制打包元数据与源码并安装包（零三方依赖，构建无需联网）。
COPY pyproject.toml README.md ./
COPY nanopore_align ./nanopore_align
RUN python -m pip install --no-cache-dir --no-build-isolation .

# 测试与 verify 所需的其余仓库内容。
COPY tests ./tests
COPY verify ./verify
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=5 \
    CMD python scripts/healthcheck.py

CMD ["sh", "-c", "python -m nanopore_align.app --host 0.0.0.0 --port ${PORT}"]
