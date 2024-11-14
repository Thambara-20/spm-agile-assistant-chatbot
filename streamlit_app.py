import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Pinecone and Gemini API configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = "us-east-1"  # replace with your environment if needed
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API key
genai.configure(api_key=gemini_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'process-guide-index'
myindex = pc.Index(index_name)

# Initialize embedding model locally
embed_model = SentenceTransformer("all-mpnet-base-v2")

# Function to generate response using Gemini
def generate_response_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Retrieve relevant passage from Pinecone
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

# Generate answer based on context and history using Gemini
async def generate_answer(query, context, history):
    history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    input_text = f"You are an Agile Assistant. Here’s the conversation so far and additional context to help answer the user's question.\n\nHistory:\n{history_text}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer as clearly and concisely as possible."
    response = generate_response_with_gemini(input_text)
    return response.strip()

# Streamlit UI
st.title("Agile Assistant Chatbot")

# Initialize session state for history if not already initialized
if "history" not in st.session_state:
    st.session_state.history = []

# Display the chat history to the user
for q, a in st.session_state.history:
    st.write(f"**User:** {q}")
    st.write(f"**Bot:** {a}")

# Input from user
query = st.text_input("Ask your question:")

if st.button("Submit") and query:
    # Get relevant context from Pinecone
    relevant_text = asyncio.run(get_relevant_passage(query))
    
    if relevant_text == "No relevant results found":
        st.write("No relevant context was found. Please ask another question.")
    else:
        # Generate the bot's response using the history and context
        answer = asyncio.run(generate_answer(query, relevant_text, st.session_state.history))
        st.write(f"**Bot:** {answer}")

        # Append the current question and answer to the history
        st.session_state.history.append((query, answer))
        
# second one ---------------------------------------------------------------------------------------------------------------------------------------------
# import os
# import streamlit as st
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# from langchain.vectorstores import Pinecone as LangchainPinecone
# from langchain.embeddings import SentenceTransformerEmbeddings
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer

# # Load environment variables
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = 'scrum-dataset-index'

# # Check if the index exists; if not, create it
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=768,  # Adjust dimension based on your embedding model
#         metric='cosine',  # Use the desired similarity metric
#         spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
#     )

# # Connect to the index
# myindex = pc.Index(index_name)

# # Initialize embedding model locally
# embed_model = SentenceTransformer("all-mpnet-base-v2")

# # Initialize local text generation model using GPT-2
# generator = pipeline("text-generation", model="gpt2-medium")

# # Function to generate embeddings locally
# def get_embedding(text):
#     return embed_model.encode(text).tolist()

# # Function to generate an answer using local GPT-2 model
# def generate_answer(system_message, prompt):
#     full_prompt = f"{system_message}\n\nUser: {prompt}\nAssistant:"
#     response = generator(full_prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)
#     return response[0]["generated_text"]

# # Function to retrieve relevant passage from Pinecone
# def get_relevant_passage(query):
#     query_embedding = get_embedding(query)
#     results = myindex.query(vector=query_embedding, top_k=1, include_metadata=True)
#     if results and results["matches"]:
#         return results["matches"][0]["metadata"].get("text", "No relevant results found.")
#     return "No relevant results found."

# # Define the system message
# system_message = (
#     "If a query lacks a direct answer, generate a response based on related features. "
#     "You are a helpful assistant answering queries relevant only to the agile dataset. "
#     "Answer questions politely. If the user asks about anything outside of the dataset, reply with: "
#     "'I can only provide answers related to the dataset, sir.'"
# )

# # Streamlit app
# st.title("Agile Assistant Chatbot")
# st.write("Welcome to the Agile Assistant Chatbot. Ask your questions below:")

# query = st.text_input("Ask your question here:")
# if query:
#     relevant_text = get_relevant_passage(query)
#     prompt = f"Query: {query}\n\nContext:\n{relevant_text}\n\nAnswer:"
#     answer = generate_answer(system_message, prompt)
#     st.write("Answer:", answer)
