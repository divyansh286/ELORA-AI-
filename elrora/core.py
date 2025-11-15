
# ============================================================
#   E L O R A   —   FULL CLEAN RAG + CHAT + OCR + SD SYSTEM
#   Updated 
# ============================================================

import os
import uuid
import torch
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import pytesseract
from gtts import gTTS
from langdetect import detect

# ===== LangChain latest imports =====
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import pdfplumber
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# UI
import gradio as gr

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = "/tmp/elora_rag"
os.makedirs(TMP_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_GEN_MODEL = "gpt2"
SD_MODEL = "runwayml/stable-diffusion-v1-5"

# ------------------------------------------------------------------------------------
# LOAD EMBEDDINGS
# ------------------------------------------------------------------------------------
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# ------------------------------------------------------------------------------------
# LOAD BETTER HF GENERATOR (clean decoding)
# ------------------------------------------------------------------------------------
hf_tokenizer = AutoTokenizer.from_pretrained(HF_GEN_MODEL, use_fast=True)
hf_model = AutoModelForCausalLM.from_pretrained(HF_GEN_MODEL).to(device)
hf_model.eval()

def generate_reply(prompt: str,
                   max_new_tokens: int = 200,
                   temperature: float = 0.7,
                   top_p: float = 0.9,
                   do_sample: bool = True,
                   stop_tokens=None):

    inputs = hf_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    output = hf_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=3,
        pad_token_id=hf_tokenizer.eos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id
    )

    decoded = hf_tokenizer.decode(output[0], skip_special_tokens=True)

    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()

    if stop_tokens:
        for st in stop_tokens:
            idx = decoded.find(st)
            if idx != -1:
                decoded = decoded[:idx].strip()

    return decoded.strip() if decoded.strip() else "I’m not sure how to respond — try rephrasing."


# ------------------------------------------------------------------------------------
# STABLE DIFFUSION
# ------------------------------------------------------------------------------------
sd_pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ------------------------------------------------------------------------------------
# SIMPLE RAG INDEX CLASS
# ------------------------------------------------------------------------------------
INDEX_FILE = os.path.join(TMP_DIR, "faiss_index")

class SimpleRagIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.texts = []
        self.metas = []

    def save(self):
        if self.index:
            faiss.write_index(self.index, INDEX_FILE + ".faiss")
        payload = {"texts": self.texts, "metas": self.metas}
        with open(INDEX_FILE + ".meta.pkl", "wb") as f:
            pickle.dump(payload, f)

    def load(self):
        meta = INDEX_FILE + ".meta.pkl"
        idx = INDEX_FILE + ".faiss"
        if os.path.exists(meta) and os.path.exists(idx):
            with open(meta, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.metas = data["metas"]
            self.index = faiss.read_index(idx)
            return True
        return False

    def add_texts(self, texts, metas):
        embs = self.embedder.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embs)

        if self.index is None:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embs)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query, k=4):
        if self.index is None:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for s, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], self.metas[idx], float(s)))
        return results

rag_index = SimpleRagIndex(embedder)
rag_index.load()

# ------------------------------------------------------------------------------------
# OCR IMPROVED PREPROCESSING
# ------------------------------------------------------------------------------------
def preprocess_for_ocr(path):
    img = cv2.imread(path)

    if img is None:
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th


def ocr_image_to_text(image, preprocess=True, lang='eng'):
    if isinstance(image, str):
        path = image
    else:
        try:
            path = image.name
        except:
            tmp = f"{TMP_DIR}/ocr_{uuid.uuid4().hex}.png"
            image.save(tmp)
            path = tmp

    if preprocess:
        try:
            proc = preprocess_for_ocr(path)
            text = pytesseract.image_to_string(proc, config='--oem 3 --psm 6')
        except:
            text = pytesseract.image_to_string(Image.open(path))
    else:
        text = pytesseract.image_to_string(Image.open(path))

    return text.strip()


def perform_ocr(image, summarize=False, preprocess=True):
    text = ocr_image_to_text(image, preprocess=preprocess)
    if not text:
        return "No text detected."

    try:
        lang = detect(text[:50])
    except:
        lang = "unknown"

    return f"Detected language: **{lang}**\n\n{text}"


# ------------------------------------------------------------------------------------
# RAG ANSWER
# ------------------------------------------------------------------------------------
def rag_answer(question):
    hits = rag_index.search(question, k=4)
    if not hits:
        # fallback to plain chat
        prompt = f"User: {question}\nElora:"
        return generate_reply(prompt)

    context = "\n\n".join([t for t, _, _ in hits])

    prompt = (
        f"You are Elora. Use ONLY the context below to answer.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nAnswer:"
    )

    return generate_reply(prompt, stop_tokens=["\nUser:"])


# ------------------------------------------------------------------------------------
# CHAT FUNCTION (REPLACED + CLEAN)
# ------------------------------------------------------------------------------------
def detect_intent(text):
    t = text.lower()
    if any(x in t for x in ["generate image", "draw", "make picture", "image of"]):
        return "image"
    if any(x in t for x in ["ocr", "extract text", "read text"]):
        return "ocr"
    if t.startswith("search my docs:") or t.startswith("rag:"):
        return "rag"
    return "chat"


def elora_chat(message, history):
    intent = detect_intent(message)

    if intent == "image":
        return "Use the **Image Generation** tab for artwork."
    if intent == "ocr":
        return "Use the **OCR Reader** tab."

    if intent == "rag":
        q = re.sub(r"^(search my docs:|rag:)\s*", "", message, flags=re.I)
        return rag_answer(q)

    # build trimmed history
    convo = []
    for u, a in history[-3:]:
        convo.append(f"User: {u}\nElora: {a}")
    prompt = "\n".join(convo) + f"\nUser: {message}\nElora:"

    return generate_reply(prompt, stop_tokens=["\nUser:", "\nElora:"])


# ------------------------------------------------------------------------------------
# INDEXER (ONLY PDF/TXT BY DEFAULT)
# ------------------------------------------------------------------------------------
def extract_text_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def index_uploaded_files(files, index_images=False):
    texts = []
    metas = []

    for file in files:
        path = file.name if hasattr(file, "name") else file
        ext = path.lower().split(".")[-1]

        if ext == "pdf":
            raw = extract_text_from_pdf(path)
        elif ext in ("txt", "md"):
            raw = open(path, "r", encoding="utf-8", errors="ignore").read()
        elif ext in ("png", "jpg", "jpeg") and index_images:
            raw = ocr_image_to_text(path)
        else:
            continue

        if not raw.strip():
            continue

        paras = [c.strip() for c in re.split(r"\n{2,}", raw) if c.strip()]
        for p in paras:
            if len(p) > 1000:
                for i in range(0, len(p), 800):
                    texts.append(p[i:i+800])
                    metas.append({"source": os.path.basename(path)})
            else:
                texts.append(p)
                metas.append({"source": os.path.basename(path)})

    if texts:
        rag_index.add_texts(texts, metas)
        rag_index.save()

    return f"Indexed {len(texts)} chunks."


# ------------------------------------------------------------------------------------
# IMAGE GENERATION
# ------------------------------------------------------------------------------------
def generate_image(prompt, style, negative, steps, scale, size):
    full_prompt = f"{prompt}, {style} style"
    img = sd_pipe(
        full_prompt,
        negative_prompt=negative,
        guidance_scale=scale,
        num_inference_steps=steps,
        height=size,
        width=size
    ).images[0]

    out_path = f"{TMP_DIR}/img_{uuid.uuid4().hex}.png"
    img.save(out_path)
    return out_path


# ------------------------------------------------------------------------------------
# TTS
# ------------------------------------------------------------------------------------
def tts_response(text, tone):
    voice_prefix = {
        "calm": "Soft tone: ",
        "energetic": "Energetic tone: ",
        "narrative": "Storytelling: "
    }.get(tone, "")

    path = f"{TMP_DIR}/tts_{uuid.uuid4().hex}.mp3"
    gTTS(voice_prefix + text).save(path)
    return path


# ------------------------------------------------------------------------------------
# UI START
# ------------------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
    gr.Markdown("# 🌌 **Elora AI — Multimodal Assistant (Fixed + Enhanced)**")

    # CHAT TAB
    with gr.Tab("💬 Chat"):
        chatbox = gr.Chatbot()
        msg = gr.Textbox(label="Your message")
        history = gr.State([])

        def chat_submit(m, h):
            reply = elora_chat(m, h)
            h.append((m, reply))
            return h, ""

        msg.submit(chat_submit, [msg, history], [chatbox, msg])
        gr.Button("Send").click(chat_submit, [msg, history], [chatbox, msg])

    # RAG TAB
    with gr.Tab("📚 Knowledge Base"):
        upload = gr.File(file_count="multiple")
        index_images = gr.Checkbox(label="Index images too (OCR)", value=False)
        btn = gr.Button("Index")
        out = gr.Textbox()
        btn.click(index_uploaded_files, [upload, index_images], out)

    # IMAGE GEN
    with gr.Tab("🎨 Image"):
        p = gr.Textbox(label="Prompt")
        st = gr.Dropdown(["realistic", "anime", "digital art"], value="realistic")
        neg = gr.Textbox(label="Negative prompt")
        steps = gr.Slider(20,80,40)
        scale = gr.Slider(5,15,7)
        size = gr.Dropdown([512,768], value=512)
        btn = gr.Button("Generate")
        img = gr.Image()
        btn.click(generate_image, [p,st,neg,steps,scale,size], img)

    # OCR TAB
    with gr.Tab("🧾 OCR"):
        img_in = gr.Image(type="filepath")
        btn = gr.Button("Extract")
        text_out = gr.Textbox()
        btn.click(perform_ocr, img_in, text_out)

    # TTS
    with gr.Tab("🔊 TTS"):
        txt = gr.Textbox(label="Text")
        tone = gr.Dropdown(["neutral","calm","energetic","narrative"])
        speak = gr.Button("Speak")
        audio = gr.Audio()
        speak.click(tts_response, [txt, tone], audio)

ui.launch(debug=False, share=True)
