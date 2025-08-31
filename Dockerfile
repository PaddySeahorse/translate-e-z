FROM python:3.11-slim

WORKDIR /app

# 基础依赖（加速编译/安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

# 拷贝代码
COPY app.py .
# 你需要把转换好的 CT2 模型目录（如 opus-mt-en-zh/）在部署时挂载或COPY进镜像：
# COPY opus-mt-en-zh /app/opus-mt-en-zh

# 环境变量（可在运行时覆盖）
ENV MODEL_DIR=/app/opus-mt-en-zh
ENV HF_TOKENIZER=Helsinki-NLP/opus-mt-en-zh
ENV MAX_BATCH_SIZE=64

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
