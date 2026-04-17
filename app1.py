# MUST BE AT THE VERY TOP
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from groq import Groq
from fastembed import TextEmbedding
import os
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
GROQ_API_KEY = "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"
COLLECTION_NAME = "academic_books"

# Initialize Groq
client = Groq(api_key=GROQ_API_KEY)

# THE SQLITE FOLDER FIX
# We create the folder structure Chroma expects on the fly
current_dir = os.path.dirname(os.path.abspath(__file__))
db_folder = os.path.join(current_dir, "research_vector_db")

if not os.path.exists(db_folder):
    os.makedirs(db_folder)
    # Move the uploaded sqlite file into the new folder
    source_file = os.path.join(current_dir, "chroma.sqlite3")
    if os.path.exists(source_file):
        shutil.copy(source_file, os.path.join(db_folder, "chroma.sqlite3"))

# Connect to the database
chroma_client = chromadb.PersistentClient(path=db_folder)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

@st.cache_resource
def load_models():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

embedding_model = load_models()

# ==========================================
# 2. CHAT ENGINE
# ==========================================
def get_ai_response(user_query, chat_history):
    query_vector = list(embedding_model.embed([user_query]))[0].tolist()

    results = collection.query(query_embeddings=[query_vector], n_results=10)
    
    context_text = ""
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        book_title = results['metadatas'][0][i].get('book_title', 'Unknown Source')
        context_text += f"\n--- Source: {book_title} ---\n{text}\n"

    system_prompt = f"""You are an elite research assistant. 
    Use the provided context to provide a detailed, analytical answer.
    Cite sources inline using [Source: Book Title].
    
    Context:\n{context_text}"""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=1500
    )
    return chat_completion.choices[0].message.content

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Academic AI", page_icon="🏛️")
st.title("🏛️ Academic Research Portal")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing library..."):
            try:
                answer = get_ai_response(user_input, st.session_state.messages[:-1])
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")