# 1. SQLITE FIX (MUST BE FIRST)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import gradio as gr
import chromadb
from groq import Groq
from fastembed import TextEmbedding
import os
import shutil

# ==========================================
# 2. CONFIGURATION (USING SECRETS/ENV)
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
COLLECTION_NAME = "academic_books"

# Initialize Groq
if not GROQ_API_KEY:
    # We use a placeholder for local testing, but HF will use the secret
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# THE SQLITE FOLDER FIX
current_dir = os.path.dirname(os.path.abspath(__file__))
db_folder = os.path.join(current_dir, "research_vector_db")

if not os.path.exists(db_folder):
    os.makedirs(db_folder)
    source_file = os.path.join(current_dir, "chroma.sqlite3")
    if os.path.exists(source_file):
        shutil.copy(source_file, os.path.join(db_folder, "chroma.sqlite3"))

# Database Initialization
chroma_client = chromadb.PersistentClient(path=db_folder)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Load Embedding Model
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

# ==========================================
# 3. RESEARCH ENGINE LOGIC
# ==========================================
def get_ai_response(message, history):
    if not client:
        return "Error: GROQ_API_KEY not found in environment secrets."

    # Convert question to vector
    query_vector = list(embedding_model.embed([message]))[0].tolist()
    
    # Search Database
    results = collection.query(query_embeddings=[query_vector], n_results=10)
    
    context_text = ""
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        book_title = results['metadatas'][0][i].get('book_title', 'Unknown Source')
        context_text += f"\n--- Source: {book_title} ---\n{text}\n"

    # Construct Prompt
    system_prompt = f"""You are an elite research assistant. 
    Use the provided context to provide a detailed, analytical answer.
    Cite sources inline using [Source: Book Title].
    Context:\n{context_text}"""
    
    # Convert Gradio history to Groq format
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Request Error: {str(e)}"

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
view = gr.ChatInterface(
    fn=get_ai_response,
    title="🏛️ Academic Research Portal",
    description="Query your research library using Llama 3.3 and local Vector Search.",
    examples=["Who was Josephine Butler?", "Summarize the Acts mentioned in the texts."],
    theme="soft"
)

if __name__ == "__main__":
    view.launch()
