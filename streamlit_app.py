import streamlit as st
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from process_save_doc import process_and_save


# Streamlit UI
st.title("WebChatMate")
st.subheader("Your Conversational URL Companion")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

url_list = []

url = st.text_input("Enter a URL:")

if st.button("Submit URL"):
    if url:
        url_list.append(url)

    vectorStore_openAI = process_and_save(url_list)
    
    llm=OpenAI(temperature=0)
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
    
    user_question = st.text_input("Enter your question:")
    
    st.write(user_question)
            