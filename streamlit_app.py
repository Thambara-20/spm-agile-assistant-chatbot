import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pinecone import Pinecone
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = "us-east-1"  
gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)

pc = Pinecone(api_key=pinecone_api_key)
index_name = 'process-guide-index'
myindex = pc.Index(index_name)

embed_model = SentenceTransformer("all-mpnet-base-v2")

def generate_response_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

async def get_relevant_passage(query):
    query_embedding = embed_model.encode(query).tolist()
    search_results = myindex.query(vector=query_embedding, top_k=3, include_metadata=True)

    if search_results.matches:
        relevant_entries = " ".join(
            f"{match.metadata.get('input', '')} {match.metadata.get('output', '')} {match.metadata.get('text', '')}"
            for match in search_results.matches
        )
        return relevant_entries
    return "No relevant results found."

async def generate_answer(query, context, history):
    history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    input_text = f"You are an Agile Assistant. Here’s the conversation so far and additional context to help answer the user's question.\n\nHistory:\n{history_text}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer as clearly and concisely as possible."
    response = generate_response_with_gemini(input_text)
    return response.strip()

st.title("Agile Assistant Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

for q, a in st.session_state.history:
    st.write(f"**You** {q}")
    st.write(f"{a}")

query = st.text_input("Ask your question:")

if st.button("Submit") and query:
    relevant_text = asyncio.run(get_relevant_passage(query))
    
    if relevant_text == "No relevant results found":
        st.write("No relevant context was found. Please ask another question.")
    else:
        answer = asyncio.run(generate_answer(query, relevant_text, st.session_state.history))
        st.write(f"{answer}")

        st.session_state.history.append((query, answer))
        
