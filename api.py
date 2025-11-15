from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

from elora.services import (
    chat_with_elora,
    generate_image_from_prompt,
    ocr_from_image_bytes,
)

app = FastAPI(
    title="Elora AI API",
    version="1.1",
    description="Multimodal RAG + Chat + OCR + ImageGen API for Elora"
)

# ---------------------------
# REQUEST MODELS
# ---------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User message")
    history: Optional[List[Tuple[str, str]]] = Field(
        default_factory=list,
        description="List of (user, assistant) message pairs"
    )


class ImgRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    style: str = "realistic"
    negative: str = ""
    steps: int = 40
    scale: float = 7.0
    size: int = 512


# ---------------------------
# API ENDPOINTS
# ---------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint for Elora.
    Takes a prompt + conversation history and returns a reply.
    """
    try:
        reply = chat_with_elora(req.prompt, req.history)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-image")
async def generate_image(req: ImgRequest):
    """
    Generates an image from a text prompt.
    Uses default params unless custom ones are provided.
    """
    try:
        path = generate_image_from_prompt(
            prompt=req.prompt,
            style=req.style,
            negative=req.negative,
            steps=req.steps,
            scale=req.scale,
            size=req.size
        )
        return {"image_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Extracts text from an uploaded image using OCR.
    """
    try:
        img_bytes = await file.read()
        text = ocr_from_image_bytes(img_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """
    Health check endpoint for load balancers / monitoring.
    """
    return {"status": "ok", "service": "Elora AI", "version": "1.1"}
