from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ctranslate2
import sentencepiece as spm
import os

app = FastAPI()
translator = ctranslate2.Translator("opus-mt-en-zh", device="cpu")  # Vercel 无 GPU
sp = spm.SentencePieceProcessor("opus-mt-en-zh/spm.model")

class TranslationRequest(BaseModel):
    texts: list[str]

@app.post("/")
async def translate(req: TranslationRequest):
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 数组不能为空")
        tokenized = [sp.encode(text, out_type=str) for text in req.texts]
        translations = translator.translate_batch(tokenized)
        decoded = [sp.decode(t[0]["tokens"]) for t in translations]
        return {"translations": decoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))