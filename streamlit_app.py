import os
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Initialize session state
if "url_list" not in st.session_state:
    st.session_state.url_list = []
    st.session_state.VectorStore = None
    st.session_state.chain = None

input_counter = 0

def get_input_key():
    global input_counter
    key = f"input_{input_counter}"
    input_counter += 1
    return key

def get_button_key():
    global input_counter
    key = f"button_{input_counter}"
    input_counter += 1
    return key

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


st.session_state.is_exiting = False
st.session_state.current_question = None

if st.session_state.chain:
    while not st.session_state.is_exiting:
        if st.session_state.current_question is None:
            st.session_state.current_question = st.text_input("Enter your question:", key=get_input_key())
            ask = st.button("Ask", key=get_button_key())
        else:
            st.write("Question:", st.session_state.current_question)
            response = st.session_state.chain(
                {"question": st.session_state.current_question}, return_only_outputs=True
            )
            answer = response["answer"].replace("\n", "")
            st.write("Answer:", answer)
            st.session_state.current_question = None