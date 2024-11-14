import os
import streamlit as st
from dotenv import load_dotenv
import pinecone  # Import Pinecone correctly
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Initialize Pinecone with your API key and environment
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")  # Ensure environment matches your Pinecone setup
index_name = 'agile-dataset-index'
myindex = pinecone.Index(index_name)  # Initialize Pinecone index

# Initialize embedding and generative models
embed_model = SentenceTransformer('all-mpnet-base-v2')
gen_model_name = "distilgpt2"
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# Function to get the relevant passage from Pinecone based on a query
def get_relevant_passage(query):
    query_embedding = embed_model.encode(query).tolist()  # Convert ndarray to list for compatibility
    results = myindex.query(vector=query_embedding, top_k=1, include_metadata=True)
    if results['matches']:
        metadata = results['matches'][0]['metadata']
        context = f"Text: {metadata.get('text', 'No text available')}"
        return context
    return "No relevant results found"

# Function to generate a response from the model based on the query and context
def generate_answer(system_message, chat_history, prompt):
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + f"\nUser: {prompt}\nAssistant:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = gen_model.generate(inputs['input_ids'], max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_history.append(f"Assistant: {response}")
    return response

# Initialize chat history and system message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

system_message = (
    "If a query lacks a direct answer, generate a response based on related features. "
    "You are a helpful assistant answering queries relevant only to the agile dataset. "
    "Answer questions politely. If the user asks about anything outside of the dataset, reply with: "
    "'I can only provide answers related to the dataset, sir.'"
)

# Streamlit UI
st.title("Agile Assistant Chatbot")

query = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if query:
        relevant_text = get_relevant_passage(query)
        prompt = f"Query: {query}\n\nContext:\n{relevant_text}\n\nAnswer:"
        answer = generate_answer(system_message, st.session_state.chat_history, prompt)
        st.write("Answer:", answer)

        # Display chat history
        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)
