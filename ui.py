import gradio as gr
from elora.services import chat_with_elora, generate_image_from_prompt, ocr_from_image_bytes

# Wrapper for OCR in Gradio (since it expects file paths)
def ocr_file(path):
    with open(path, "rb") as f:
        return ocr_from_image_bytes(f.read())

with gr.Blocks() as elora_ui:
    gr.Markdown("## 🧠 Elora AI - Assistant | Text ✍️ + Image 🎨 + OCR 📖")

    with gr.Tab("🗣️ Chatbot"):
        gr.ChatInterface(fn=chat_with_elora, title="Talk to Elora")

    with gr.Tab("🎨 Image Generator"):
        prompt = gr.Textbox(label="Enter prompt")
        gen_btn = gr.Button("Generate")
        image_out = gr.Image(label="Generated Image")
        gen_btn.click(fn=generate_image_from_prompt, inputs=prompt, outputs=image_out)

    with gr.Tab("📖 OCR Reader"):
        ocr_input = gr.Image(type="filepath", label="Upload Image")
        ocr_btn = gr.Button("Read Text")
        ocr_output = gr.Textbox(label="OCR Result")
        ocr_btn.click(fn=ocr_file, inputs=ocr_input, outputs=ocr_output)

if __name__ == "__main__":
    elora_ui.launch()
