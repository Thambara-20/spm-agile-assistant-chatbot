# import os
# import streamlit as st
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load environment variables
# load_dotenv()

# # Configure Pinecone
# api_key = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=api_key)
# index_name = 'scrum-dataset-index'
# myindex = pc.Index(index_name)

# # Initialize embedding and generative models
# embed_model = SentenceTransformer('all-mpnet-base-v2')
# gen_model_name = "gpt2-medium"  # Using a larger model for better response quality
# gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
# tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# # Function to get the relevant passage from Pinecone based on a query
# def get_relevant_passage(query):
#     query_embedding = embed_model.encode(query).tolist()  # Convert ndarray to list for compatibility
#     results = myindex.query(vector=query_embedding, top_k=1, include_metadata=True)
#     if results['matches']:
#         metadata = results['matches'][0]['metadata']
#         context = f"Text: {metadata.get('text', 'No text available')}"
#         return context
#     return "No relevant results found"

# # Function to generate a response from the model based on the query and context
# def generate_answer(system_message, prompt):
#     # Simplify the prompt to focus on the latest query and context
#     full_prompt = f"{system_message}\n\nUser: {prompt}\nAssistant:"
    
#     # Generate response with controlled settings
#     inputs = tokenizer(full_prompt, return_tensors="pt")
#     outputs = gen_model.generate(
#         inputs['input_ids'],
#         max_length=100,
#         temperature=0.7,
#         top_p=0.9,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return response

# # Initialize chat history and system message
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# system_message = (
#     "If a query lacks a direct answer, generate a response based on related features. "
#     "You are a helpful assistant answering queries relevant only to the agile dataset. "
#     "Answer questions politely. If the user asks about anything outside of the dataset, reply with: "
#     "'I can only provide answers related to the dataset, sir.'"
# )

# # Streamlit UI
# st.title("Agile Assistant Chatbot")

# query = st.text_input("Ask your question:")

# if st.button("Get Answer"):
#     if query:
#         # Retrieve relevant passage from Pinecone and create a prompt
#         relevant_text = get_relevant_passage(query)
#         prompt = f"Query: {query}\n\nContext:\n{relevant_text}\n\nAnswer:"
        
#         # Generate and display the final answer
#         answer = generate_answer(system_message, prompt)
#         st.write("Answer:", answer)

#         # Store chat history
#         st.session_state.chat_history.append(f"User: {query}")
#         st.session_state.chat_history.append(f"Assistant: {answer}")

#         # Display chat history
#         with st.expander("Chat History"):
#             for chat in st.session_state.chat_history:
#                 st.write(chat)

# second one ---------------------------------------------------------------------------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
import requests
import pinecone

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'scrum-dataset-index'
myindex = pc.Index(index_name)
embed_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vector_store = Pinecone(index_name=index_name, embedding_function=embed_model)

# Function to get embeddings from Hugging Face API
def get_embedding(text, api_key):
    url = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, json={"inputs": text})
    return response.json()[0]  # Assuming the embedding is returned as a list

# Function to generate a response from Hugging Face API
def generate_answer(system_message, prompt, api_key):
    full_prompt = f"{system_message}\n\nUser: {prompt}\nAssistant:"
    url = "https://api-inference.huggingface.co/models/gpt2-medium"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, json={"inputs": full_prompt, "parameters": {"max_length": 100}})
    
    # Ensure the response is in the expected format
    try:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            return "Error: Unexpected response format"
    except (KeyError, IndexError, TypeError):
        return "Error generating response"

# Function to retrieve relevant passage from Pinecone
def get_relevant_passage(query):
    query_embedding = get_embedding(query, hf_api_key)
    results = vector_store.similarity_search(query_embedding, k=1)
    if results:
        return results[0].metadata.get("text", "No relevant results found.")
    return "No relevant results found."

# Define the system message
system_message = (
    "If a query lacks a direct answer, generate a response based on related features. "
    "You are a helpful assistant answering queries relevant only to the agile dataset. "
    "Answer questions politely. If the user asks about anything outside of the dataset, reply with: "
    "'I can only provide answers related to the dataset, sir.'"
)

# Streamlit UI
st.title("Agile Assistant Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("Ask your question:")
if st.button("Get Answer") and query:
    # Retrieve relevant passage and create a prompt
    relevant_text = get_relevant_passage(query)
    prompt = f"Query: {query}\n\nContext:\n{relevant_text}\n\nAnswer:"
    
    # Generate the answer and display
    answer = generate_answer(system_message, prompt, hf_api_key)
    st.write("Answer:", answer)

    # Update chat history
    st.session_state.chat_history.append(f"User: {query}")
    st.session_state.chat_history.append(f"Assistant: {answer}")

# Display chat history
with st.expander("Chat History"):
    for chat in st.session_state.chat_history:
        st.write(chat)

