# Agile Assistant Chatbot
URL: [https://agile-ai.streamlit.app/](https://agile-ai.streamlit.app/)
## Overview
Theis is an AI assistant built using **Retrieval-Augmented Generation (RAG)** to provide intelligent and context-aware answers to Agile-related queries. The bot combines **Pinecone** for efficient vector-based search, **Sentence Transformers** for generating text embeddings, and **Google Gemini** for generating human-like answers. It is designed to help users with Agile practices, Scrum methodologies, and other related topics by leveraging a custom Agile dataset.

By using **RAG**, the chatbot enhances its ability to answer questions by retrieving relevant context from a large dataset and then generating accurate, natural-sounding responses.

---

## Key Features
- **Context-Aware Conversations**: Maintains conversation history and answers queries based on previous interactions and relevant context.
- **Pinecone Vector Search**: Uses **Pinecone** to search for relevant documents from a custom Agile dataset based on the user's query.
- **Google Gemini Generative AI**: Generates fluent, natural language responses based on the context retrieved.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval and generative models to provide more accurate answers.
- **Streamlit Interface**: Interactive web interface built using **Streamlit**, where users can input their questions and get answers in real time.

---

## Requirements

### Python Libraries
You will need to install the following Python libraries to run the application:

```bash
pip install streamlit python-dotenv langchain sentence-transformers google-generativeai pinecone-client datasets

```
You want to create a .env file and set ```GEMINI_API_KEY``` and ```PINECONE_API_KEY``` accordingly.
