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


def process_and_save(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=300)
    docs = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)
    
    return vectorStore_openAI