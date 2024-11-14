# Agile Assistant Chatbot

## Overview
This AI-powered assistant is here to help you with your Agile-related questions. Whether you’re looking for answers on Scrum, Agile processes, or best practices, the chatbot is designed to offer relevant, context-aware responses. Powered by **Pinecone**, **Sentence Transformers**, and **Google Gemini** (Generative AI), this tool provides a seamless experience with answers based on your input and a pre-defined Agile dataset.

### Key Features
- **Context-Aware Conversations**: Keeps track of the conversation to offer responses based on your previous questions.
- **Intelligent Search**: Uses **Pinecone** to search through relevant documents, ensuring answers are tailored to your query.
- **Natural Language Generation**: Generates human-like responses using **Google Gemini AI**.
- **Interactive Interface**: Built using **Streamlit**, making it easy to ask questions and get responses instantly.

---

## Requirements

### Python Libraries
To get started, you'll need to install a few Python libraries. You can do this easily by running the following command:

```bash
pip install streamlit python-dotenv langchain sentence-transformers google-generativeai pinecone-client
