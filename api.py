from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from elora.services import chat_with_elora, generate_image_from_prompt, ocr_from_image_bytes

app = FastAPI(title="Elora AI API", version="1.0")

# Request model for chat
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chat endpoint - send a prompt + optional history, get AI reply.
    """
    return {"reply": chat_with_elora(req.prompt, req.history)}

# Request model for image generation
class ImgRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(req: ImgRequest):
    """
    Image generation endpoint - send a text prompt, get generated image path.
    """
    path = generate_image_from_prompt(req.prompt)
    return {"image_path": path}

@app.post("/ocr")
def ocr(file: UploadFile = File(...)):
    """
    OCR endpoint - upload an image file, get extracted text.
    """
    img_bytes = file.file.read()
    return {"text": ocr_from_image_bytes(img_bytes)}

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
