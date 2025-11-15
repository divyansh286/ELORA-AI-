import gradio as gr
from elora.services import (
    chat_with_elora,
    generate_image_from_prompt,
    ocr_from_image_bytes
)

# -------------------------------------------------------------------
# Helper for OCR from file path
# -------------------------------------------------------------------
def ocr_file(path):
    try:
        with open(path, "rb") as f:
            return ocr_from_image_bytes(f.read())
    except Exception as e:
        return f"⚠️ OCR Error: {e}"


# -------------------------------------------------------------------
# Custom Chat Handler for Better History
# -------------------------------------------------------------------
def chat_wrapper(message, history):
    # Convert Gradio's history format [(u1,a1),(u2,a2), ...]
    # into the required list
    formatted_history = [(u, a) for u, a in history]
    reply = chat_with_elora(message, formatted_history)
    return reply


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as elora_ui:

    gr.Markdown(
        """
        # 🌌 **Elora AI — Multimodal Intelligence**
        Smarter Chat 💬 | Image Generation 🎨 | OCR 📖  
        ---
        """,
    )

    # ---------------------------
    #  CHAT TAB
    # ---------------------------
    with gr.Tab("🧠 Chat"):

        gr.Markdown(
            """
            #### Talk to Elora  
            Ask questions, chat casually, or reference indexed documents.
            """
        )

        gr.ChatInterface(
            fn=chat_wrapper,
            title="Elora Chat",
            chatbot=gr.Chatbot(height=450),
            textbox=gr.Textbox(placeholder="Ask Elora anything...", container=True),
            clear_btn="Clear",
        )

    # ---------------------------
    #  IMAGE GENERATION TAB
    # ---------------------------
    with gr.Tab("🎨 Image Generator"):

        gr.Markdown(
            """
            ### Create Artwork  
            Generate images with Stable Diffusion using natural language.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Example: A futuristic samurai standing on neon rooftops...",
                )

                style = gr.Dropdown(
                    ["realistic", "anime", "digital art", "painting"],
                    value="realistic",
                    label="Style",
                )

                negative = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Optional: blur, low quality, distorted face...",
                )

                steps = gr.Slider(10, 80, value=40, step=1, label="Inference Steps")
                scale = gr.Slider(1, 15, value=7, step=0.5, label="Guidance Scale")
                size = gr.Dropdown([512, 768], value=512, label="Image Size")

                gen_btn = gr.Button("🎨 Generate Image", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Generated Image", height=450)

        def wrapper_gen(p, st, neg, s, sc, sz):
            return generate_image_from_prompt(
                prompt=p,
                style=st,
                negative=neg,
                steps=s,
                scale=sc,
                size=sz
            )

        gen_btn.click(
            fn=wrapper_gen,
            inputs=[prompt, style, negative, steps, scale, size],
            outputs=image_out
        )

    # ---------------------------
    #  OCR TAB
    # ---------------------------
    with gr.Tab("📖 OCR Reader"):

        gr.Markdown(
            """
            ### Extract Text from Images  
            Upload any document, screenshot, scanned page, or image containing text.
            """
        )

        ocr_input = gr.Image(type="filepath", label="Upload Image", height=350)
        ocr_btn = gr.Button("📘 Extract Text", variant="primary")
        ocr_output = gr.Textbox(label="Extracted Text", lines=12)

        ocr_btn.click(fn=ocr_file, inputs=ocr_input, outputs=ocr_output)


# -------------------------------------------------------------------
# Launch
# -------------------------------------------------------------------
if __name__ == "__main__":
    elora_ui.launch(debug=False, share = True)
