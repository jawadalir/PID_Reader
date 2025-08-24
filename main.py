# app.py
# -*- coding: utf-8 -*-
"""
Streamlit P&ID / HAZOP Extractor with Progressive Memory & Chat History Support
"""

import os
import json
import time
import base64
import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# ---------------- Config ----------------
MODEL_DEFAULT = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ---------------- OpenAI client ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Utils ----------------
def render_pdf_page_to_png(pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.5) -> bytes:
    """Render one PDF page to PNG bytes"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        raise IndexError(f"PDF has {len(doc)} pages; page_index {page_index} is out of range.")
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    data = pix.tobytes("png")
    doc.close()
    return data

def load_user_prompts(file) -> list[str]:
    """Extract only user text prompts from chat_history.json"""
    try:
        data = json.load(file)
    except Exception as e:
        st.error(f"⚠️ Invalid chat_history.json: {e}")
        return []

    prompts = []
    for entry in data:
        if isinstance(entry, dict) and entry.get("role") == "user":
            c = entry.get("content")
            if isinstance(c, str):
                if c.strip():
                    prompts.append(c.strip())
            elif isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "").strip()
                        if t:
                            parts.append(t)
                if parts:
                    prompts.append("\n".join(parts))
    return prompts

def process_pdf(pdf_path):
    """Check if PDF has text (HAZOP) or is scanned/diagram (P&ID)"""
    doc = fitz.open(pdf_path)
    text_content = ""
    for page in doc:
        text_content += page.get_text()
    doc.close()

    if text_content.strip():
        return {
            "type": "text",
            "text": f"Here is the HAZOP document content:\n{text_content[:5000]}..."
        }
    else:
        return {
            "type": "image_url",
            "image_url": {"url": f"file://{os.path.abspath(pdf_path)}"}
        }

def ask_model_json(model: str, system_prompt: str, prompt_text: str, image_png_bytes: bytes = None):
    """Ask GPT with enforced JSON output"""
    content = [{"type": "text", "text": prompt_text}]
    if image_png_bytes:
        b64 = base64.b64encode(image_png_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {"role": "user", "content": content}

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_DELAY * attempt)
    raise last_exc


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="P&ID / HAZOP Extractor", layout="wide")
st.title("🛠️ P&ID / HAZOP Extractor with Progressive Memory & Chat History")

uploaded_pdf = st.file_uploader("📂 Upload your PDF (P&ID or HAZOP)", type=["pdf"])
uploaded_history = st.file_uploader("📝 Upload chat_history.json", type=["json"])
page_num = st.number_input("Page number (1-based)", min_value=1, value=1, step=1)
zoom = st.slider("Zoom level", 1.0, 4.0, 2.5, 0.5)

if uploaded_pdf and uploaded_history:
    # Save uploaded PDF temporarily for processing
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    # Detect PDF type
    pdf_info = process_pdf(temp_pdf_path)

    # Render if diagram
    png_bytes = None
    if pdf_info["type"] == "image_url":
        try:
            with open(temp_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            png_bytes = render_pdf_page_to_png(pdf_bytes, page_index=page_num - 1, zoom=zoom)
            st.image(png_bytes, caption=f"Page {page_num}", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not render page: {e}")
            st.stop()

    prompts = load_user_prompts(uploaded_history)
    if not prompts:
        st.warning("⚠️ No valid user prompts found in chat_history.json.")
        st.stop()

    memory = []
    progress = st.progress(0, text="Starting analysis...")

    # ---- Phase 1: progressive memory ----
    system_prompt = (
        "You are an expert process engineer. Read the provided P&ID/HAZOP document and accumulated memory. "
        "For the current user prompt, update the JSON knowledge base. "
        "Always return { 'pipelines': [...], 'instruments': [...] }. "
        "Keep consistency with previous memory."
    )

    for i, prompt in enumerate(prompts, start=1):
        context = {"previous_memory": memory, "current_prompt": prompt}
        progress.progress((i-1)/len(prompts), text=f"Processing prompt {i}/{len(prompts)}...")

        try:
            if pdf_info["type"] == "text":
                # Pass extracted text instead of image
                result = ask_model_json(MODEL_DEFAULT, system_prompt, json.dumps(context, ensure_ascii=False) + "\n" + pdf_info["text"])
            else:
                # Pass diagram as PNG
                result = ask_model_json(MODEL_DEFAULT, system_prompt, json.dumps(context, ensure_ascii=False), png_bytes)

            entry = {"prompt": prompt, "reply": result}
            memory.append(entry)

            st.subheader(f"🔍 Prompt {i}: {prompt}")
            st.json(result)
            st.toast(f"Prompt {i} processed ✅")
        except Exception as e:
            st.error(f"⚠️ Failed prompt {i}: {e}")

    # ---- Phase 2: final consolidation ----
    if memory:
        final_prompt = (
            "You are a process engineer. Using the document and full memory of prompts+replies, "
            "create one final JSON with ALL pipelines and instruments. "
            "Output only { 'pipelines': [...], 'instruments': [...] }."
        )
        progress.progress(1.0, text="Final consolidation...")
        try:
            final_input = json.dumps(memory, ensure_ascii=False)
            if pdf_info["type"] == "text":
                final_json = ask_model_json(MODEL_DEFAULT, final_prompt, final_input + "\n" + pdf_info["text"])
            else:
                final_json = ask_model_json(MODEL_DEFAULT, final_prompt, final_input, png_bytes)

            st.success("🎉 Final consolidated JSON generated!")
            st.subheader("📊 Final Result")
            st.json(final_json)

            st.download_button("📥 Download final.json", json.dumps(final_json, indent=2),
                               file_name="final.json", mime="application/json")
        except Exception as e:
            st.error(f"⚠️ Final step failed: {e}")

    progress.empty()

