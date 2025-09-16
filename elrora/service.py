from .core import elora_chat, generate_image, perform_ocr
from PIL import Image
import io

def chat_with_elora(prompt: str, history: list = None) -> str:
    """
    Wrapper for chatbot function.
    """
    return elora_chat(prompt, history or [])

def generate_image_from_prompt(prompt: str) -> str:
    """
    Wrapper for image generation function.
    Returns the file path of the generated image.
    """
    return generate_image(prompt)

def ocr_from_image_bytes(img_bytes: bytes) -> str:
    """
    Wrapper for OCR function.
    Accepts raw image bytes and returns extracted text.
    """
    img = Image.open(io.BytesIO(img_bytes))
    return perform_ocr(img)
