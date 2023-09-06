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

# Initialize session state
st.session_state.url_list = []
st.session_state.VectorStore = None
st.session_state.chain = None

st.title("WebChatMate")
st.subheader("Your Conversational URL Companion")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

url = st.text_input("Enter Your URL")

submit = st.button('Submit')

if submit and url:
    
    st.session_state.url_list.append(url)
    st.session_state.VectorStore = process_and_save(st.session_state.url_list)
    llm = OpenAI(temperature=0)
    st.session_state.chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.VectorStore.as_retriever())

    user_question = st.text_input("Enter your question:")

    ask = st.button('Ask')
    
    if ask and user_question and st.session_state.chain:
    
        response = st.session_state.chain({"question": user_question}, return_only_outputs=True)
        answer = response['answer'].replace('\n', '')
        sources = response.get('sources', '')
        st.write("Answer:", answer)
        st.write("Sources:", sources)
            
 
            


