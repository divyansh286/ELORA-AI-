
# 🧠 Elora AI Chatbot

**Elora AI** is a multi-modal, intelligent assistant that merges the power of conversational AI, image generation, and OCR technology. Built to be fast, functional, and creator-customizable, Elora is your complete personal assistant — right in your browser.

# 🧠 Elora AI — Multimodal Assistant

Elora is a multimodal GenAI-powered assistant that combines:
- 🤖 **LLM chatbot** (DistilGPT2)
- 🎨 **Image generation** (Stable Diffusion v1-4)
- 📖 **OCR text extraction** (Tesseract)
- 🎤 **Speech synthesis** (gTTS)

## 🚀 Run with FastAPI

## bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000


---

## 🚀 Features

- 🗣️ **Conversational AI** with DistilGPT2  
- 🎨 **Text-to-Image Generation** using Stable Diffusion v1-4  
- 👁️ **OCR** (Image-to-Text) with Tesseract  
- 🔊 **Text-to-Speech** using gTTS (future-ready)  
- 🌐 **Interactive Web UI** via Gradio  
- 🛡️ **Creator Authentication** and custom rule engine  
- 📁 **Document and Image Upload Support**  
- 💬 **Real-Time Interaction** with flexible natural language commands  

---

## ⚙️ Models Used

| Module           | Model                     | Purpose                       |
|------------------|---------------------------|-------------------------------|
| Chatbot          | `DistilGPT2`              | Conversational AI             |
| Image Generation | `Stable Diffusion v1-4`   | Text-to-Image Synthesis       |
| OCR              | `Tesseract` (via `pytesseract`) | Image Text Extraction    |
| TTS (Optional)   | `gTTS`                    | Voice Output (Text-to-Speech) |

---

## 🌐 Interface Modes

- **Chat Mode** – Talk with Elora in natural language  
- **Generate Mode** – Create artwork from text prompts  
- **Read Mode** – Extract text from uploaded images  

---

## 🌟 Workflow Overview

```mermaid
graph TD
    A[fast_api_User Interface] --> B{Choose Mode}
    
    B --> |Chat Mode| C[DistilGPT2]
    B --> |Generate Mode| D[Stable Diffusion v1-4]
    B --> |Read Mode| E[Tesseract OCR]
    
    C --> F[Response Generated]
    D --> G[Image Created]
    E --> H[Text Extracted]
    
    F --> I[Display Answer]
    G --> J[Show Artwork]
    H --> K[Reveal Text]

    style A fill:#FF6B6B,stroke:#333
    style B fill:#4ECDC4,stroke:#333
    style C fill:#45B7D1,stroke:#333
    style D fill:#96CEB4,stroke:#333
    style E fill:#FFEEAD,stroke:#333
    style F fill:#ffffff,stroke:#DAA520
    style G fill:#ffffff,stroke:#DAA520
    style H fill:#ffffff,stroke:#DAA520
    style I fill:#FF9999,stroke:#FF4500
    style J fill:#FF9999,stroke:#FF4500
    style K fill:#FF9999,stroke:#FF4500
