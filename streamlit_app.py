import os

import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def process_and_save(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=300
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    return vectorStore_openAI


if "url_list" not in st.session_state:
    # Initialize session state
    st.session_state.url_list = []
    st.session_state.VectorStore = None
    st.session_state.chain = None

st.title("WebChatMate")
st.subheader("Your Conversational URL Companion")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

url = st.text_input("Enter Your URL")

submit = st.button("Submit")

if submit and url:
    if url not in st.session_state.url_list:
        st.session_state.url_list += [url]
    st.session_state.VectorStore = process_and_save(st.session_state.url_list)
    llm = OpenAI(temperature=0)
    st.session_state.chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=st.session_state.VectorStore.as_retriever()
    )

st.write("URLs", st.session_state.url_list)

if st.session_state.chain:
    user_question = st.text_input("Enter your question:")

    ask = st.button("Ask")

    if ask and user_question and st.session_state.chain:
        response = st.session_state.chain(
            {"question": user_question}, return_only_outputs=True
        )
        answer = response["answer"].replace("\n", "")
        sources = response.get("sources", "")
        st.write("Answer:", answer)
        st.write("Sources:", sources)