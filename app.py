import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS




# Load your OpenAI key
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # Replace with your key

PDF_PATH = "document.pdf"

@st.cache_data
def load_pdf_text():
    with fitz.open(PDF_PATH) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

@st.cache_resource
def load_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

@st.cache_resource
def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit UI
st.title("📄 PDF Chatbot (No GCP)")
st.write("Ask any question from the uploaded document!")

if not os.path.exists(PDF_PATH):
    st.error("document.pdf not found! Upload it to the project folder.")
    st.stop()

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Processing..."):
        text = load_pdf_text()
        vectordb = load_vector_store(text)
        qa_chain = get_qa_chain(vectordb)
        answer = qa_chain.run(query)
        st.write("💬 Answer:")
        st.success(answer)
