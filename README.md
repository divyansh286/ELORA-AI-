
# 🤖 ELORA AI Assistant v3

A multi-modal AI assistant combining text generation, image generation, and OCR with a sleek GPT-style interface. Fully optimized for Colab.

---

## 🧠 Models Used

### 1. 🗣️ Chatbot: **DistilGPT2**
- **Purpose**: Conversational AI / Text Generation  
- **Architecture**: Distilled GPT-2 (6-layer transformer, 82M parameters)  
- **Benefits**:
  - 97% of GPT-2’s performance
  - 40% smaller and faster  
- **Implementation**:
  - Tokenizer and causal language modeling via `transformers` (Hugging Face)

---

### 2. 🎨 Image Generation: **Stable Diffusion v1-4**
- **Purpose**: Text-to-image synthesis  
- **Model**: Latent Diffusion Model (~890M params)  
- **Features**:
  - 512x512 output
  - CLIP guidance for quality
  - FP16 inference (faster, GPU-optimized)  
- **Library**: `diffusers`

---

### 3. 👁️ OCR Engine: **Tesseract via pytesseract**
- **Purpose**: Extract text from images  
- **Features**:
  - Multi-language support
  - Handles real-world image input  
- **Library**: `pytesseract`

---

### 4. 🔊 Text-to-Speech: **gTTS**
- Integrated for voice replies (planned for full deployment)

---

### 5. 🖼️ UI/UX Layer: **Gradio**
- Clean GPT-style web UI
- Mode selector (Chat / Generate / OCR)

---

## 🔁 Workflow Overview

```mermaid
graph TD
    A[🎪 User Interface] --> B{🔀 Choose Mode}
    
    B --> |🗣️ Chat Mode| C[🧠 DistilGPT2]
    B --> |🎨 Generate Mode| D[🖌️ Stable Diffusion v1-4]
    B --> |👁️ Read Mode| E[📸 Tesseract OCR]
    
    C --> F[💬 Response Generated]
    D --> G[🌌 Image Created]
    E --> H[📜 Text Extracted]
    
    F --> I[🎉 Display Answer]
    G --> J[🖼️ Show Artwork]
    H --> K[🔍 Reveal Text]
    
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

