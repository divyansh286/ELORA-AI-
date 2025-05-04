

# Install all required packages
!pip install -q transformers diffusers gradio accelerate pytesseract gtts

# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import pytesseract
from gtts import gTTS
from IPython.display import Audio
import gradio as gr
import uuid
import os

# Load chatbot model (DistilGPT2 for light weight)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16 if device=="cuda" else torch.float32)
pipe = pipe.to(device)

# Chatbot function compatible with gr.ChatInterface
def elora_chat(message, history):
    input_text = message
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Image generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    filename = f"/tmp/gen_{uuid.uuid4().hex}.png"
    image.save(filename)
    return filename

# OCR function
def perform_ocr(image):
    if isinstance(image, str):
        image = Image.open(image)
    text = pytesseract.image_to_string(image)
    return text if text.strip() else "No text detected."

# Gradio UI
with gr.Blocks() as elora_ui:
    gr.Markdown("## 🧠 Elora AI - Assistant | Text ✍️ + Image 🎨 + OCR 📖")

    with gr.Tab("🗣️ Chatbot"):
        gr.ChatInterface(fn=elora_chat, title="Talk to Elora")

    with gr.Tab("🎨 Image Generator"):
        prompt = gr.Textbox(label="Enter prompt")
        gen_btn = gr.Button("Generate")
        image_out = gr.Image(label="Generated Image")
        gen_btn.click(fn=generate_image, inputs=prompt, outputs=image_out)

    with gr.Tab("📖 OCR Reader"):
        ocr_input = gr.Image(type="filepath", label="Upload Image")
        ocr_btn = gr.Button("Read Text")
        ocr_output = gr.Textbox(label="OCR Result")
        ocr_btn.click(fn=perform_ocr, inputs=ocr_input, outputs=ocr_output)

elora_ui.launch(debug=False)

