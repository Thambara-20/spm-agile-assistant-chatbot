import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone with your API key and environment
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'process-guide-index'

# Connect to the index
myindex = pc.Index(index_name)

# Initialize models with try-except blocks to isolate errors
try:
    # Embedding model initialization
    embed_model = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")

gen_model_name = "distilgpt2"
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# Set eos_token as pad_token to avoid padding error
tokenizer.pad_token = tokenizer.eos_token

# Function to get the relevant passage from Pinecone based on a query
def get_relevant_passage(query):
    try:
        query_embedding = embed_model.encode(query).tolist()  # Convert ndarray to list for compatibility
        results = myindex.query(vector=query_embedding, top_k=1, include_metadata=True)
        if results['matches']:
            metadata = results['matches'][0]['metadata']
            context = metadata.get('text', 'No text available')
            return context
        return "No relevant results found"
    except Exception as e:
        st.error(f"Error retrieving passage from Pinecone: {e}")
        return "No relevant results found"

# Function to generate a response from the model based on the query and context
def generate_answer(query, context):
    try:
        # Simplified and clarified prompt
        full_prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        # Tokenize and generate response
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
        outputs = gen_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # Generate up to 50 new tokens
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Fallback for repetitive or meaningless responses
        if response.lower().count("answer") > 5 or len(response) < 10:
            return "I'm sorry, I couldn't generate a meaningful response. Please try asking in a different way."
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Unable to generate a response."

# Initialize chat history and system message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

system_message = (
    "You are a helpful assistant answering queries relevant only to the Agile dataset. "
    "If a query lacks a direct answer, generate a response based on related features. "
    "Answer questions politely. If the user asks about anything outside of the dataset, reply with: "
    "'I can only provide answers related to the dataset, sir.'"
)

# Streamlit UI
st.title("Agile Assistant Chatbot")

query = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if query:
        # Retrieve relevant passage from Pinecone
        relevant_text = get_relevant_passage(query)

        # Generate and display the final answer
        if relevant_text == "No relevant results found":
            answer = "I'm sorry, I couldn't find any relevant information in the dataset."
        else:
            answer = generate_answer(query, relevant_text)
        
        st.write("Answer:", answer)

        # Store chat history
        st.session_state.chat_history.append(f"User: {query}")
        st.session_state.chat_history.append(f"Assistant: {answer}")

        # Display chat history
        with st.expander("Chat History"):
            st.write(system_message)  # Display the system message as context in the history
            for chat in st.session_state.chat_history:
                st.write(chat)
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
