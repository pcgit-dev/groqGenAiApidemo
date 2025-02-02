import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq 
import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
import time

load_dotenv()

#Load groq api key
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ['GROQ_API_KEY'] = groq_api_key

llm = ChatGroq(groq_api_key=groq_api_key,model="llama3-8b-8192")
promt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided contex only .
    Please provide the most accurate response based on the question .
    <context>
         {context}
    </context>
    Question : {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
        st.session_state.final_documents= st.session_state.text_splitter.split_documents()
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter your query from the researc paper")

if st.button("Document Emebedding"):
    create_vector_embedding()
    st.write("Vector data base is ready")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,promt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time-start}")
    st.write(response['answer'])

    with st.expander("Document Similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
