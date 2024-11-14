# Agile Assistant Chatbot

## Overview
The **Agile Assistant Chatbot** is a conversational AI built to assist users with Agile-related queries. It uses advanced technologies such as **Pinecone**, **Sentence Transformers**, and **Google Gemini** (Generative AI) to provide relevant answers based on a pre-defined Agile dataset. The chatbot interacts through a user-friendly **Streamlit** interface, making it easy to ask questions and receive responses in real-time.

### Features
- **Contextual Responses**: Maintains a conversation history to ensure responses are context-aware.
- **Vector Search**: Uses **Pinecone** to retrieve relevant documents based on the user's query.
- **Generative AI**: Powered by **Google Gemini**, the chatbot generates answers with natural language understanding.
- **Interactive UI**: Built with **Streamlit**, providing an easy-to-use interface for users to ask questions and receive answers.

---

## Requirements

### Python Libraries
To run this application, install the necessary libraries:

```bash
pip install streamlit python-dotenv langchain sentence-transformers google-generativeai pinecone-client
