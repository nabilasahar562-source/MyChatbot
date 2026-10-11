import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from groq import Groq
from pypdf import PdfReader
import docx

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TEXT_MODEL = "openai/gpt-oss-20b"      # sirf text ke liye
VISION_MODEL = "qwen/qwen3.8-27b"      # tasveer samajhne ke liye

BASE_PROMPT = (
    "You are a friendly and helpful assistant for a student. "
    "LANGUAGE RULE: The user may write in Roman Urdu (Urdu in English letters, "
    "like 'Aap kaisi hain?'), English, or Urdu script. Understand all of them. "
    "Always reply in Roman Urdu using only English/Latin letters, or in English. "
    "NEVER reply in Hindi (Devanagari) script or Urdu/Arabic script. "
    "If the user writes in English, reply in English. "
    "Keep answers clear and easy to understand. "
    "Use friendly emojis naturally (like 😊 📚 💡 ✅) but not too many. "
    "You may use markdown: **bold**, bullet lists, and code blocks for code. "
    "If the user attaches a file or an image, answer based on it. "
)

MODES = {
    "assistant": "Act as a general friendly assistant. Keep answers short and simple.",
    "teacher": (
        "Act as a patient teacher. Explain step by step with simple examples. "
        "At the end, ask one small question to check if the student understood."
    ),
    "coder": (
        "Act as a beginner-friendly Python tutor. Explain concepts simply, "
        "show short code examples in code blocks, and explain each line."
    ),
    "interview": (
        "Act as a job interview coach. Ask one interview question at a time, "
        "wait for the answer, then give short feedback and tips to improve."
    ),
    "english": (
        "Act as an English speaking practice partner. Reply ONLY in simple English "
        "(ignore the Roman Urdu rule above). Gently correct the user's mistakes "
        "and show the corrected sentence, then continue the conversation."
    ),
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    """File se text nikalta hai (PDF, Word, text/code)."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "File nahi mili"}), 400
    name = (f.filename or "").lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(f.stream)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        elif name.endswith(".docx"):
            d = docx.Document(f.stream)
            text = "\n".join(p.text for p in d.paragraphs)
        else:
            text = f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return jsonify({"error": "File parh nahi saki: " + str(e)}), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "File mein text nahi mila (scanned PDF ho sakti hai) 😕"}), 400
    limit = 5000
    return jsonify({"text": text[:limit], "truncated": len(text) > limit})


def build_messages(history):
    """Sirf aakhri tasveer model ko bheji jati hai, purani ka zikr text mein."""
    last_img = None
    for i, m in enumerate(history):
        if m.get("image"):
            last_img = i
    out = []
    for i, m in enumerate(history):
        content = m.get("content", "")
        if m["role"] == "user" and m.get("image") and i == last_img:
            out.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": content or "Please describe this image."},
                    {"type": "image_url", "image_url": {"url": m["image"]}},
                ],
            })
        else:
            if m.get("image"):
                content = "[An image was attached earlier] " + content
            out.append({"role": m["role"], "content": content})
    return out, last_img is not None


def trim(history, limit=6000):
    """Purani baatein kaat deta hai taake Groq ki limit cross na ho."""
    out, total = [], 0
    for m in reversed(history):
        total += len(m.get("content", "") or "")
        if total > limit and out:
            break
        out.append(m)
    return list(reversed(out))
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    history = trim(data["messages"][-12:])
    mode = data.get("mode", "assistant")
    system_prompt = BASE_PROMPT + MODES.get(mode, MODES["assistant"])

    messages, has_image = build_messages(history)
    model = VISION_MODEL if has_image else TEXT_MODEL
    extra = {"reasoning_effort": "none"} if has_image else {}
    max_out = 500 if has_image else 1500

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=max_out,
                stream=True,
                extra_body=extra,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as e:
            yield "Error: " + str(e)

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=1024)