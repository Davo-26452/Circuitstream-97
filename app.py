import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

import chromadb
from doc_helper import read_file

load_dotenv()

db= chromadb.PersistentClient(path="./chroma_db")
brain = db.get_or_create_collection("documents")
memory = db.get_or_create_collection("conversations")

def chunk_it(text,size=400):
    bits = text.split(". ")
    chunks, current = [], ""
    for bit in bits:
        if len(current) + len(bit) < size:
            current += bit + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = bit + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def store_document(file):
    chunks = chunk_it (read_file(file))
    prefix = file.name.replace(" ", "_")
    brain.upsert(
        documents=chunks,
        ids=[f"{prefix}_chunk{i}" for i in range(len(chunks))],
    )
    return len(chunks)
def store_conversation(file):

    st.title("Stormai")

if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.header('Settings')
    name=st.text_input('David-Paul')
    mood=st.selectbox("What mood will your Ai have", ["Happy","Sad","Angry"])
    creativity = st.slider("Creativity", 0.0,1.0,0.3)
    model=st.selectbox("Model"["openai/gpt-oss-120b", "openai/gpt-oss-20b" ])

prompt = st.chat_input("What can i help you with today")

user_input = st.chat_input("How are you", accept_file=True, file_type=["pdf", "txt"])

if user_input:
    prompt = user_input.text
    if user_input.files:
        with st.spinner:

if user_input and prompt:
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        r = client.chat.completions.create(
            model=model,
            temperature=creativity,
            messages=[{"role":"user","content": prompt}],
        )
    st.write(r)

