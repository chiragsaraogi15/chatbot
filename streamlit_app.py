import streamlit as st
import os
import pickle
import faiss
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# Streamlit UI
st.title("WebChatMate: Your Conversational URL Companion")

# Define the path where you want to save the file
file_path = "faiss_store_openai.pkl"

# Initialize variables to store the vector store and LLM chain
VectorStore = None
chain = None

def create_or_load_vector_store(docs):
    if not os.path.exists(file_path):
        embeddings = OpenAIEmbeddings()
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorStore_openAI, f)
    else:
        with open(file_path, "rb") as f:
            vectorStore_openAI = pickle.load(f)
    return vectorStore_openAI
    
url = st.text_input("Enter URL:")

if st.button("Submit"):
    try:
        # Load and preprocess the data from the URL
        urls = [url]
        
        loaders = UnstructuredURLLoader(urls=urls)
        
        data = loaders.load()

        text_splitter = CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=300)
        
        docs = text_splitter.split_documents(data)

        # Create or load the FAISS vector store
        VectorStore = create_or_load_vector_store(docs)

        # Initialize the LLM and QA chain
        llm = OpenAI(temperature=0)
        
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
question = st.text_input("Enter your question:")

if st.button("Ask"):
    try:
        if chain is not None:
            response = chain({"question": question}, return_only_outputs=True)

            # Get the answer from the response and remove the newline character
            answer = response['answer'].replace('\n', '')

            # Preserve the 'sources' information
            sources = response.get('sources', '')

            # Display the answer and sources
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Sources:")
            st.write(sources)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Reset button
if st.button("Reset"):
    st.text_input("Enter URL:")
    st.text_input("Enter your question:")

# Exit button
if st.button("Exit"):
    st.stop()