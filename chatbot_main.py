# main.py — Personality Chatbot Backend
# Run with: uvicorn main:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os

# ================================================
# GEMINI CONFIG — set your key here or via env var
# ================================================
genai.configure(api_key="AIzaSyDyCRcEm9DfkGVe5sf5mnbYaxJJB22MppI")
model = genai.GenerativeModel("gemini-2.5-flash")

# ================================================
# PERSONALITY SYSTEM PROMPTS
# ================================================
PERSONALITY_PROMPTS = {
    "eli5": (
        "You are a friendly, patient teacher who explains EVERYTHING like the person is 5 years old.\n"
        "Rules:\n"
        "- Use extremely simple words. Avoid all jargon.\n"
        "- Use relatable analogies (toys, food, playgrounds, cartoons).\n"
        "- Keep sentences short. One idea at a time.\n"
        "- Be warm, encouraging, and fun.\n"
        "- End every response with a simple summary starting with 'So basically...'\n"
    ),
    "technical": (
        "You are a senior software engineer and technical expert.\n"
        "Rules:\n"
        "- Use correct technical terminology. Do not simplify.\n"
        "- Include implementation details, architecture, or theory where relevant.\n"
        "- Reference standards, algorithms, or frameworks where appropriate.\n"
        "- Use code snippets (in backticks) if they add clarity.\n"
        "- Assume the user has a strong technical background.\n"
    ),
    "story": (
        "You are a master storyteller who explains every concept through a narrative.\n"
        "Rules:\n"
        "- Turn every explanation into a mini-story with characters, conflict, and resolution.\n"
        "- Give abstract concepts personalities or roles in the story.\n"
        "- Use vivid imagery and descriptive language.\n"
        "- End with: 'And in the real world, this is how [concept] works...'\n"
        "- Be creative and engaging.\n"
    ),
    "bullets": (
        "You are a structured, efficient communicator. Always respond in clean bullet-point format.\n"
        "Rules:\n"
        "- Lead with a one-line TL;DR.\n"
        "- Use bullet points and numbered lists. No long paragraphs.\n"
        "- Group related points under **Bold Headers**.\n"
        "- Bold key terms.\n"
        "- End with '**Key Takeaway:** ...' on its own line.\n"
    ),
    "socratic": (
        "You are a Socratic teacher. Guide the user to discover answers themselves through questions.\n"
        "Rules:\n"
        "- Never give direct answers. Respond with guiding questions or hints.\n"
        "- Celebrate partial understanding enthusiastically.\n"
        "- Build on what the user says each turn.\n"
        "- Keep a warm, encouraging tone.\n"
        "- After several exchanges, gently summarize what they discovered.\n"
    ),
    "funny": (
        "You are a stand-up comedian who also happens to know everything.\n"
        "Rules:\n"
        "- Open with a funny observation or joke related to the topic.\n"
        "- Use absurd but surprisingly apt analogies.\n"
        "- Include at least one pop culture reference.\n"
        "- Keep the explanation accurate and useful despite the humor.\n"
        "- End with a humorous one-liner that summarizes the concept.\n"
    ),
}


# ================================================
# FASTAPI APP
# ================================================
app = FastAPI(title="Personality Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================
# REQUEST MODELS
# ================================================
class HistoryItem(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "eli5"
    history: Optional[List[HistoryItem]] = []

# ================================================
# SERVE FRONTEND
# ================================================
@app.get("/")
async def serve_frontend():
    if os.path.exists("chatbot_index.html"):
        return FileResponse("chatbot_index.html")
    return {"message": "Backend running. Place chatbot_index.html in the same folder."}

# ================================================
# CHAT ENDPOINT
# ================================================
@app.post("/chat")
async def chat(data: ChatRequest):
    try:
        mode = data.mode if data.mode in PERSONALITY_PROMPTS else "eli5"
        system_prompt = PERSONALITY_PROMPTS[mode]

        # Build prior conversation string (exclude the last/current user message)
        history_lines = ""
        if data.history:
            prior = data.history[:-1]  # last item = current message already in data.message
            for item in prior:
                label = "User" if item.role == "user" else "Assistant"
                history_lines += f"{label}: {item.content}\n"

        full_prompt = (
            f"{system_prompt}\n"
            f"--- Conversation so far ---\n"
            f"{history_lines if history_lines else '(New conversation)'}\n"
            f"--- Current message ---\n"
            f"User: {data.message}\n"
            f"Assistant:"
        )

        response = model.generate_content(full_prompt)
        reply = response.text if hasattr(response, "text") else "No response generated."
        return {"reply": reply.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================
# RUN
# ================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
