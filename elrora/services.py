from .core import elora_chat, generate_image, perform_ocr
from PIL import Image
import io

# -------------------------------
# CHAT WRAPPER
# -------------------------------
def chat_with_elora(prompt: str, history: list = None) -> str:
    """
    Wrapper for Elora chat function.
    'history' should be a list of (user_message, bot_reply) tuples.
    """
    return elora_chat(prompt, history or [])


# -------------------------------
# IMAGE GENERATION WRAPPER
# -------------------------------
def generate_image_from_prompt(
    prompt: str,
    style: str = "realistic",
    negative: str = "",
    steps: int = 40,
    scale: float = 7.0,
    size: int = 512
) -> str:
    """
    Wrapper around generate_image from core.
    Provides defaults so external apps don't need UI parameters.
    Returns filepath to the generated image.
    """
    return generate_image(prompt, style, negative, steps, scale, size)


# -------------------------------
# OCR WRAPPER
# -------------------------------
def ocr_from_image_bytes(img_bytes: bytes, summarize: bool = False, preprocess: bool = True) -> str:
    """
    Wrapper for OCR function.
    Accepts raw image bytes and returns extracted text.
    """
    img = Image.open(io.BytesIO(img_bytes))
    return perform_ocr(img, summarize=summarize, preprocess=preprocess)

