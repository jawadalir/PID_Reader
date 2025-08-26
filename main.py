# app.py
# -*- coding: utf-8 -*-
"""
Streamlit P&ID / HAZOP Extractor — updated UI
- Same functionality as before
- UI improvements:
  • Chat bubbles styled (user right, assistant left)
  • Chat input pinned at bottom (uses on_change callback to avoid session_state error)
  • Attractive sidebar + main "Exports" panel with working JSON downloads
  • Single assistant response per user turn (no duplicates)
"""

import os
import json
import time
import base64
import traceback
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# ---------------- Config ----------------
MODEL_DEFAULT = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o")
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CHAT_MAX_TOKENS = 800
PDF_PREVIEW_DPI = 144
CHAT_HISTORY_KEEP = 40
CHAT_HISTORY_FILE = "history_new.json"  # local file path

# ---------------- OpenAI client ----------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.set_page_config(page_title="P&ID / HAZOP Extractor", layout="wide")
    st.error("⚠️ OPENAI_API_KEY environment variable is not set.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
/* Subtle app polish */
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1rem; }

/* Panel headings */
h2, h3 { letter-spacing: .2px; }

/* Sidebar cards */
.sidebar-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 10px;
}

/* Chat container */
.chat-box {
    max-height: 420px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 12px;
    background: #fafafa;
    border: 1px solid #e5e7eb;
}

/* User bubble */
.user-bubble {
    background: #DCF8C6;
    padding: 10px 12px;
    border-radius: 14px;
    margin: 8px 0;
    text-align: right;
    max-width: 78%;
    margin-left: auto;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    line-height: 1.45;
    word-wrap: break-word;
}

/* Assistant bubble */
.assistant-bubble {
    background: #ffffff;
    padding: 10px 12px;
    border-radius: 14px;
    margin: 8px 0;
    text-align: left;
    max-width: 78%;
    margin-right: auto;
    border: 1px solid #f0f0f0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    line-height: 1.5;
    word-wrap: break-word;
}

/* Fixed chat input */
.chat-input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 8px 0 2px 0;
    border-top: 1px solid #e5e7eb;
}

/* Align the text_input to look like a bar */
.chat-input .stTextInput>div>div input {
    border-radius: 9999px;
    padding: 10px 16px;
}

/* Small caption for exports */
.exports-caption {
    color: #6b7280;
    font-size: 0.88rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Utils ----------------
def render_pdf_page_to_png(pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0) -> Optional[bytes]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return None
        page = doc[page_index]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        data = pix.tobytes("png")
        doc.close()
        return data
    except Exception:
        return None

def load_user_prompts_from_local(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"⚠️ Could not read local chat_history.json: {e}")
        return []
    prompts: List[str] = []

    def collect(node):
        if isinstance(node, dict):
            role = node.get("role")
            content = node.get("content")
            if role == "user":
                if isinstance(content, str) and content.strip():
                    prompts.append(content.strip())
                elif isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text", "").strip()
                            if t:
                                parts.append(t)
                        elif isinstance(part, str) and part.strip():
                            parts.append(part.strip())
                    if parts:
                        prompts.append("\n".join(parts))
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(data)
    seen, unique = set(), []
    for p in prompts:
        if p and p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for p in doc:
            try:
                pages_text.append(p.get_text())
            except Exception:
                pages_text.append("")
        doc.close()
        all_text = "\n".join(pages_text).strip()
        if all_text:
            return {"type": "text", "text": all_text[:50000]}
        else:
            return {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(pdf_path)}"}}
    except Exception:
        return {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(pdf_path)}"}}

def safe_parse_json_string(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        return {"_raw": s[:3000]}

def ask_model_json(model: str, system_prompt: str, prompt_text: str, image_png_bytes: bytes = None, max_tokens: int = 3000) -> Dict[str, Any]:
    content = [{"type": "text", "text": prompt_text}]
    if image_png_bytes:
        try:
            b64 = base64.b64encode(image_png_bytes).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        except Exception:
            pass
    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {"role": "user", "content": content}

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            if isinstance(raw, str):
                parsed = safe_parse_json_string(raw)
                if isinstance(parsed, dict):
                    return parsed
            return {"_raw": str(raw)[:3000]}
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"ask_model_json failed after {MAX_RETRIES} attempts: {last_exc}")

def ask_model_text(model: str, system_prompt: str, prompt_text: str, image_png_bytes: bytes = None, max_tokens: int = CHAT_MAX_TOKENS) -> str:
    content = [{"type": "text", "text": prompt_text}]
    if image_png_bytes:
        try:
            b64 = base64.b64encode(image_png_bytes).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        except Exception:
            pass
    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {"role": "user", "content": content}

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.0,
                max_tokens=max_tokens
            )
            raw = resp.choices[0].message.content or ""
            return raw if isinstance(raw, str) else str(raw)
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"ask_model_text failed after {MAX_RETRIES} attempts: {last_exc}")

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="P&ID / HAZOP Extractor", layout="wide")
st.title("🛠️ P&ID / HAZOP Extractor")

# Session state (initialize chat_input BEFORE widget creation)
initial_state = {
    "final_json": None,
    "pdf_context": "",
    "png_bytes": None,
    "processing_logs": [],
    "memory": [],
    "chat_history": [],
    "processing_done": False,
    "pdf_info": None,
    "chat_input": "",  # initialize so we can safely use it as widget key
}
for k, v in initial_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
sb = st.sidebar
sb.title("🧭 Processing & Exports")
progress_placeholder = sb.empty()
logs_placeholder = sb.container()
export_preview = sb.container()
export_buttons = sb.container()

def append_log(msg: str):
    st.session_state["processing_logs"].append(f"{time.strftime('%H:%M:%S')} — {msg}")
    st.session_state["processing_logs"] = st.session_state["processing_logs"][-200:]

def show_logs():
    with logs_placeholder:
        logs_placeholder.empty()
        with st.container():
            st.markdown("<div class='sidebar-card'><b>Recent logs</b></div>", unsafe_allow_html=True)
            for l in st.session_state.get("processing_logs", [])[-30:]:
                st.text(l)

def update_exports_display():
    final_json = st.session_state.get("final_json")
    export_preview.empty()
    export_buttons.empty()
    with export_preview:
        if final_json:
            st.markdown("<div class='sidebar-card'><b>Final JSON (preview)</b></div>", unsafe_allow_html=True)
            try:
                st.json(final_json)
            except Exception:
                st.text(json.dumps(final_json, indent=2)[:4000])
    with export_buttons:
        if final_json:
            tag = str(int(time.time()))
            st.download_button(
                "📥 Download final.json",
                json.dumps(final_json, indent=2, ensure_ascii=False),
                file_name=f"final_{tag}.json",
                mime="application/json",
                use_container_width=True
            )
            st.download_button(
                "📥 Download pdf_context.txt",
                st.session_state.get("pdf_context", ""),
                file_name=f"pdf_context_{tag}.txt",
                mime="text/plain",
                use_container_width=True
            )

show_logs()
update_exports_display()

# ---------------- PDF Upload ----------------
st.subheader("📂 Upload PDF")
uploaded_pdf = st.file_uploader("Upload P&ID or HAZOP PDF", type=["pdf"])

if uploaded_pdf:
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())
    append_log("Saved uploaded PDF.")

    pdf_info = process_pdf(temp_pdf_path)
    st.session_state["pdf_info"] = pdf_info
    append_log(f"PDF detection: type={pdf_info.get('type')}")

    if pdf_info.get("type") == "image_url":
        with open(temp_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        png_bytes = render_pdf_page_to_png(pdf_bytes)
        if png_bytes:
            st.image(png_bytes, caption="Preview (page 1)", use_column_width=True)
            st.session_state["png_bytes"] = png_bytes
    else:
        st.session_state["pdf_context"] = pdf_info.get("text", "")
        excerpt = st.session_state["pdf_context"][:2000]
        st.text_area("PDF text excerpt", excerpt, height=200)

    # Process button
    if st.button("🚀 Process PDF", use_container_width=True):
        append_log("Starting sequential processing.")
        prompts = load_user_prompts_from_local(CHAT_HISTORY_FILE)
        memory: List[Dict[str, Any]] = []

        for i, prompt in enumerate(prompts, start=1):
            frac = (i - 1) / max(1, len(prompts))
            progress_placeholder.progress(frac)
            append_log(f"Processing prompt {i}/{len(prompts)}")

            system_prompt = (
                "You are an expert process engineer. Update the JSON knowledge base from the P&ID/HAZOP. "
                "Always return { 'pipelines': [...], 'instruments': [...] }."
            )
            context = {"previous_memory": memory, "current_prompt": prompt}
            context_json = json.dumps(context, ensure_ascii=False)

            try:
                if pdf_info.get("type") == "text":
                    call_input = context_json + "\n" + st.session_state.get("pdf_context", "")[:50000]
                    result = ask_model_json(MODEL_DEFAULT, system_prompt, call_input)
                else:
                    result = ask_model_json(
                        MODEL_DEFAULT,
                        system_prompt,
                        context_json,
                        image_png_bytes=st.session_state.get("png_bytes")
                    )
            except Exception as e:
                append_log(f"Prompt {i} failed: {e}")
                memory.append({"prompt": prompt, "reply": {"_error": str(e)}})
                continue

            memory.append({"prompt": prompt, "reply": result})
            append_log(f"Prompt {i} done.")
            show_logs()

        # Final consolidation
        append_log("Starting final consolidation.")
        final_prompt = (
            "You are a process engineer. Using memory, create final JSON with ALL pipelines and instruments."
        )
        try:
            final_input = json.dumps(memory, ensure_ascii=False)
            if pdf_info.get("type") == "text":
                final_json = ask_model_json(
                    MODEL_DEFAULT,
                    final_prompt,
                    final_input + "\n" + st.session_state.get("pdf_context", "")[:50000]
                )
            else:
                final_json = ask_model_json(
                    MODEL_DEFAULT,
                    final_prompt,
                    final_input,
                    image_png_bytes=st.session_state.get("png_bytes")
                )
        except Exception as e:
            tb = traceback.format_exc()
            final_json = {"pipelines": [], "instruments": [], "_error": str(e), "_trace": tb}

        st.session_state["final_json"] = final_json
        st.session_state["memory"] = memory
        st.session_state["processing_done"] = True
        append_log("Final JSON produced.")
        progress_placeholder.progress(1.0)
        st.success("🎉 Final consolidated JSON generated!")

        # Refresh exports (sidebar)
        update_exports_display()

        # Also show a compact Exports panel in main area
        with st.expander("📦 Exports", expanded=True):
            tag = str(int(time.time()))
            st.markdown("<span class='exports-caption'>Downloads are also available in the sidebar.</span>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download final.json",
                    json.dumps(st.session_state["final_json"], indent=2, ensure_ascii=False),
                    file_name=f"final_{tag}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📥 Download pdf_context.txt",
                    st.session_state.get("pdf_context", ""),
                    file_name=f"pdf_context_{tag}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ---------------- Chatbot ----------------
st.markdown("---")
st.subheader("💬 Interactive Chat")

disabled_chat = not st.session_state.get("processing_done", False)

# Chat history in bubbles
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# === Chat submission handler (callback) ===
def _handle_chat_submit():
    """Runs when text_input with key='chat_input' changes (on_change)."""
    user_msg_local = st.session_state.get("chat_input", "").strip()
    # Clear field immediately if empty or chat disabled
    if not user_msg_local:
        st.session_state["chat_input"] = ""
        return
    if not st.session_state.get("processing_done", False):
        st.session_state["chat_input"] = ""
        # Optionally log or provide a small feedback; avoid st.warning here inside callback
        append_log("User attempted to chat before processing completed.")
        return

    # Append user message
    st.session_state["chat_history"].append({"role": "user", "content": user_msg_local})
    # Trim if too long
    if len(st.session_state["chat_history"]) > CHAT_HISTORY_KEEP:
        st.session_state["chat_history"] = st.session_state["chat_history"][-CHAT_HISTORY_KEEP:]

    # Build context
    system_prompt = """
        You are a highly skilled Process Engineering Assistant with expertise in interpreting and analyzing Piping & Instrumentation Diagrams (P&IDs). 
        - Always provide clear, structured, and technically accurate explanations. 
        - Use the final JSON or PDF data if available to support your answers. 
        - When a P&ID is provided, carefully analyze it and explain equipment, instruments, control loops, and process flow in detail. 
        - If the information is incomplete, ask clarifying questions instead of making assumptions. 
        - Keep responses concise, professional, and easy to follow for engineers and students.

        """
    context_parts = []
    if st.session_state.get("final_json"):
        context_parts.append("Final JSON:\n" + json.dumps(st.session_state["final_json"], ensure_ascii=False)[:20000])
    if st.session_state.get("pdf_context"):
        context_parts.append("PDF excerpt:\n" + st.session_state.get("pdf_context")[:20000])
    model_input = "\n\n".join(context_parts + [user_msg_local])

    # Call model (blocking) and save reply once
    try:
        # show simple spinner while waiting (works in callback)
        with st.spinner("Assistant is typing..."):
            reply_text = ask_model_text(
                CHAT_MODEL,
                system_prompt,
                model_input,
                image_png_bytes=st.session_state.get("png_bytes")
            )
    except Exception as e:
        reply_text = f"Chat error: {e}"

    st.session_state["chat_history"].append({"role": "assistant", "content": reply_text})

    # Clear the input (allowed inside callback)
    st.session_state["chat_input"] = ""

# Input pinned at bottom (uses on_change callback)
with st.container():
    st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
    st.text_input(
        "Ask about pipelines, instruments, risks...",
        key="chat_input",
        disabled=disabled_chat,
        label_visibility="collapsed",
        placeholder="Ask about pipelines, instruments, risks...",
        on_change=_handle_chat_submit
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar logs (live)
sb.divider()
sb.markdown("### Recent logs")
for l in st.session_state.get("processing_logs", [])[-12:]:
    sb.text(l)
