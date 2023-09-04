import streamlit as st
import llama_index
import openai
import os

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

# Set the page title
st.title("Document Conversation App")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Initialize the index
if uploaded_file:
    documents = llama_index.Document.from_file(uploaded_file)
    index = llama_index.VectorStoreIndex.from_documents([documents])
    index.storage_context.persist()
    storage_context = llama_index.StorageContext.from_defaults(persist_dir="./storage")
    index = llama_index.load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    st.success("PDF document uploaded and indexed successfully!")

    # Initialize a conversation history
    conversation_history = []

    # Function to generate a response using the LLM
    def generate_response(prompt):
        # Add the prompt to the conversation history
        conversation_history.append(prompt)

        # Generate a response using the entire conversation history as context
        response = query_engine.query('\n'.join(conversation_history))

        # Convert the response to a string
        llm_response = str(response)

        # Add the LLM's response to the conversation history
        conversation_history.append(llm_response)

        return llm_response

    # User input for conversation
    user_input = st.text_input("Ask a question or provide a comment:")

    if st.button("Submit"):
        if user_input:
            llm_response = generate_response(user_input)
            st.text("LLM Response:")
            st.write(llm_response)
        else:
            st.warning("Please enter a question or comment.")

    # Display conversation history
    if conversation_history:
        st.text("Conversation History:")
        st.write('\n'.join(conversation_history))
