# Resume RAG Chatbot

An AI-powered Q&A system that answers questions about 
a resume using RAG (Retrieval Augmented Generation).

## Tech Stack
Python • LangChain • OpenAI API • FAISS • Streamlit

## How it works
1. Resume is loaded and chunked into segments
2. Chunks are embedded using OpenAI embeddings
3. Stored in FAISS vector store
4. User queries are matched to relevant chunks
5. GPT generates answers grounded in resume context

## How to Run
pip install -r requirements.txt
streamlit run app.py
