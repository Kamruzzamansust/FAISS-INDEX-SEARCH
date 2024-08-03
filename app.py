import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_directory = r"D:\All_data_science_project\Langchain\Langchain_roadmap\experiment"
faiss_index_path = os.path.join(faiss_directory, "faiss_index")
new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Streamlit app
st.title("FAISS Index Search App")

# Input query from user
query = st.text_input("Enter your query:", "")

if query:
    # Perform similarity search
    searchDocs = new_db.similarity_search(query)
    
    # Collect results
    result = []
    for doc in searchDocs:
        result.append(doc.page_content)
    
    # Display results
    st.subheader("Search Results:")
    for i, res in enumerate(result):
        st.write(f"Result {i + 1}:")
        st.write(res)


