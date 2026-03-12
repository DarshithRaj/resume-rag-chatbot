import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.title("Resume RAG Chatbot")

uploaded_file = st.file_uploader("Upload your Resume", type="pdf")

if uploaded_file:

    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(docs, embeddings)

    query = st.text_input("Ask something about the resume")

    if query:

        docs = vector_db.similarity_search(query)

        context = " ".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-3.5-turbo")

        prompt = f"""
        Answer the question based on the resume below.

        Resume:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.write("### Answer")
        st.write(response.content)