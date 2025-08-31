from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import os

# ===== Config =====
MODEL_DIR = os.getenv("MODEL_DIR", "./opus-mt-en-zh")  # 你转换后的 ct2 模型目录
HF_TOKENIZER = os.getenv("HF_TOKENIZER", "Helsinki-NLP/opus-mt-en-zh")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))  # 你可以根据显存/内存调整
# ==================

app = FastAPI(title="Batch Translate API", version="1.0.0")

class TranslateIn(BaseModel):
    texts: List[str] = Field(..., description="Array of English texts")

    @validator("texts")
    def non_empty(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("texts must be a non-empty array")
        if any((t is None or not isinstance(t, str)) for t in v):
            raise ValueError("every item in texts must be a string")
        return v

class TranslateOut(BaseModel):
    translations: List[str]

# ---- Lazy load translator (CTranslate2 + HF tokenizer) ----
translator = None
tokenizer = None

def load_model():
    global translator, tokenizer
    if translator is not None and tokenizer is not None:
        return
    try:
        # pip install ctranslate2 transformers sentencepiece
        import ctranslate2
        from transformers import AutoTokenizer

        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(
                f"CT2 model dir '{MODEL_DIR}' not found. "
                f"Please set MODEL_DIR to your converted model path."
            )
        translator = ctranslate2.Translator(MODEL_DIR, device="cpu")  # or set device="auto"
        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")

def _translate_batch(texts: List[str]) -> List[str]:
    """
    Translate English -> Chinese using CTranslate2 Marian model.
    Preserves input order and length 1:1.
    """
    # Tokenize to source sentences (Marian expects tokenized lists)
    src_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t, add_special_tokens=True)) for t in texts]

    # Run translation; you can tune beam_size, max_decoding_length, etc. for speed/quality
    results = translator.translate_batch(
        src_tokens,
        beam_size=4,
        max_decoding_length=256
    )

    # Detokenize back to text
    detok: List[str] = []
    for res in results:
        tokens = res.hypotheses[0]
        # Convert tokens back to ids via tokenizer, then decode
        try:
            ids = tokenizer.convert_tokens_to_ids(tokens)
            text = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            # Fallback: join tokens (rarely needed)
            text = "".join([t.replace("▁", " ") for t in tokens]).strip()
        detok.append(text)
    return detok

@app.post("/", response_model=TranslateOut)
@app.post("/translate", response_model=TranslateOut)
async def translate(payload: TranslateIn, request: Request):
    # Validate batch size
    if len(payload.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"Batch size {len(payload.texts)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
        )

    # Load model on first request
    try:
        load_model()
    except RuntimeError as e:
        # 500 with clear error for你的脚本捕捉
        raise HTTPException(status_code=500, detail=str(e))

    # Empty strings直接返回空字符串，保证长度与顺序 1:1 对齐
    to_translate = [t if t is not None else "" for t in payload.texts]
    try:
        # 为了速度，你也可以在这里做分批（如果 texts 很大），不过我们已经限制了 MAX_BATCH_SIZE
        translations = _translate_batch(to_translate)
        # 强保证：输出长度 == 输入长度
        if len(translations) != len(payload.texts):
            raise RuntimeError("Internal error: output length mismatch")
        return TranslateOut(translations=translations)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

@app.get("/healthz")
async def health():
    return {"status": "ok", "model_dir": MODEL_DIR, "max_batch_size": MAX_BATCH_SIZE}