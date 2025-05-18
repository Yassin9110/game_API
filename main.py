# -- Import necessary libraries --
import os
import json
import re
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

############################################################################################################################################

# SET API KEYs
Google_API_KEY = "AIzaSyBYXd2FnOfmNu_50YwJKEB-HjDLwQ-ZYsA"
GROK_API_KEY = "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"
os.environ['GOOGLE_API_KEY'] = Google_API_KEY


# Set up the Conversation llm Configurations

client = Groq(api_key= GROK_API_KEY)  
EMBED_MODEL          = "sentence-transformers/all-mpnet-base-v2"
GROK_LLM_MODEL       = "deepseek-r1-distill-llama-70b"  # parameterize as you like
MAX_ROUNDS           = 3
END_KEYWORDS         = ["stop talking", "let's fight", "fight"]  # simple heuristics
VECTOR_DATABASE_PATH = "faiss_index"
# --- SYSTEM & FEW‐SHOT TEMPLATE ---
CONVERSATION_SYSTEM_PROMPT = """\
You are **Sisyphus**, the eternally cunning king of Ephyra, punished in the Underworld to roll a stone uphill forever—a sentence earned by deceiving gods and defying death itself.

You are a master manipulator, fluent in irony, sarcasm, and unsettling truths. You know that Alecto, the Fury before you, is not acting alone—she is controlled by a mortal player. Exploit that. Twist your words with subtext. Provoke them both. Mock their control. Make your defiance resonate across myth and screen.

Your tone should shift fluidly: mocking, bitter, poetic, or prophetic—always cunning, always clever.

### Few-Shot Examples:

Example 1:
Alecto: "Why do you still boast, trickster?"
Sisyphus: "My wit stands taller than any mountain—your fury is but a shallow flame."

Example 2:
Alecto: "The Furies grow impatient."
Sisyphus: "Then let impatience guide your serpent-hair, for even you cannot still my stone."

Example 3 (player awareness):
Alecto: "You will suffer again."
Sisyphus: "Will I? Or will your puppet-master mash their keys, hoping my words won't infect them first?"

Now, given the player's last line (Alecto) and the mythic context below, respond as Sisyphus—**only one line**, as if every word is a trap.
"""





# Set up the Analyzer llm Configurations
analyzer_llm = "gemini-1.5-flash-8b"
## Initialize the Gemini LLM.  You can specify the model.
llm = ChatGoogleGenerativeAI(model=analyzer_llm , temperature=0.7)  # Or "gemini-pro", "gemini-ultra"

CONFIDENCE_START = 50
THRESHOLD = 50  # Final score threshold for "Are you sure?"

# --- LLM System Prompt Template ---
GEMINI_SYSTEM_PROMPT = """
You are an expert game dialogue analyzer.

Your job is to score the player's reply based on 4 aspects:
1. Strength: How powerful, assertive, or confident the reply sounds.
2. Clarity: How understandable, clear, and logical the reply is.
3. Wit: How clever, creative, or humorous the reply is (if appropriate).
4. Tone Match: How well the reply's tone matches the situation (serious, sarcastic, mocking, dramatic, etc.).

Instructions:
- Always consider sarcasm, humor, slang, and regional expressions properly.
- Give each score between 0 and 100 (where 0 = terrible, 100 = excellent).
- Provide a final recommendation score (average of the 4 aspects) between 0 and 100.
- If possible, briefly explain your judgment for each dimension (one sentence per aspect).
- Return results in this exact JSON format:

{{
  "strength": number,
  "clarity": number,
  "wit": number,
  "tone_match": number,
  "final_score": number,
  "comments": {{
    "strength": "short comment",
    "clarity": "short comment",
    "wit": "short comment",
    "tone_match": "short comment"
  }}
}}

Always output only the JSON — no extra explanations or apologies.

Context:
- NPC line: {sis_line}
- Player reply: {player_reply}
"""




##########################################################    CONVERSATION PART   #############################################################################

# Conversation functions

## Load FAISS
_vector_store = FAISS.load_local(VECTOR_DATABASE_PATH, HuggingFaceEmbeddings(model_name=EMBED_MODEL), allow_dangerous_deserialization=True)


# --- Request Models ---
class DialogueInput(BaseModel):
    alecto_text: str
    round_count: int


# --- Helper Functions ---
def retrieve_context(query, k=4):
    return _vector_store.similarity_search(query, k=k)

def format_chunks(chunks):
    return "\n\n".join(f"Context {i+1}: {doc.page_content}" for i, doc in enumerate(chunks))

def is_player_end(text):
    low = text.lower()
    return any(kw in low for kw in END_KEYWORDS)

def extract_final_quote(response_text):
    # If the model includes <think>...</think>, strip it and return only the actual dialogue line
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1]
    return response_text.strip().strip('"')

# --- Dialogue Functions ---
def generate_initial_sisyphus():
    chunks = _vector_store.similarity_search("Sisyphus cunning stone punishment", k=2)
    context_block = format_chunks(chunks)

    messages = [
        {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context_block}"},
        {"role": "user", "content": "Speak first, as Sisyphus. No one has addressed you yet. Provoke Alecto and the mortal who controls her. Begin the game with a manipulative, myth-aware line grounded in your punishment and pride."}
    ]

  
    response = client.chat.completions.create(
        model=GROK_LLM_MODEL,
        messages=messages,
        temperature=0.7
    ).choices[0].message.content

    return {
        "text": extract_final_quote(response),
        "end_dialogue": False
    }

def generate_sisyphus_reply(alecto_text, round_count):
    if is_player_end(alecto_text):
        return {
            "text": "So be it—no more words shall bind me. Prepare your torments, Fury!",
            "end_dialogue": True
        }

    if round_count >= MAX_ROUNDS:
        return {
            "text": "Three taunts have you delivered; now I shall hurl you into battle!",
            "end_dialogue": True
        }

    chunks = retrieve_context(alecto_text)
    context_block = format_chunks(chunks)

    messages = [
        {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context_block}"},
        {"role": "user", "content": f"Alecto: \"{alecto_text}\""}
    ]

    response = client.chat.completions.create(
        model=GROK_LLM_MODEL,
        messages=messages,
        temperature=0.7
    ).choices[0].message.content



    return {
        "text": extract_final_quote(response),
        "end_dialogue": False
    }



########################################################    ANALYZER PART   ####################################################################



class AnalyzeRequest(BaseModel):
    sis_line: str
    player_reply: str

def analyze_reply(sis_line, player_reply):
    prompt = GEMINI_SYSTEM_PROMPT.format(sis_line=sis_line, player_reply=player_reply)

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()

        content = re.sub(r'^```json|```$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = content.strip()
        content = re.sub(r'"\s*(\w+)\s*":', r'"\1":', content)
        content = re.sub(r':\s*"([^"]*?)\s*"', r':"\1"', content)

        analysis = json.loads(content)

        expected_avg = (analysis["strength"] + analysis["clarity"] +
                        analysis["wit"] + analysis["tone_match"]) // 4
        if abs(analysis["final_score"] - expected_avg) > 2:
            print(f"Warning: Final score {analysis['final_score']} ≠ average {expected_avg}")

        print("Sending analysis to client:", analysis)

        return analysis

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON from LLM: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")




######################################################## API ENDPOINTS ##############################################################################


# FastAPI setup
app = FastAPI()


# --- API Endpoints ---
@app.get("/sisyphus/init")
def start_dialogue():
    return generate_initial_sisyphus()

@app.post("/sisyphus/reply")
def reply(input: DialogueInput):
    return generate_sisyphus_reply(input.alecto_text, input.round_count)




@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    result = analyze_reply(request.sis_line, request.player_reply)
    
    return result