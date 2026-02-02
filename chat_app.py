import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
PERSIST_DIR = "chroma_store"
EMBED_MODEL = "hkunlp/instructor-base"

#run - python -m streamlit run chat_app.py
# pip list | findstr langchain
# Not Required
# langchain                                1.0.3
# langchain-classic                        1.0.0

#Required
# langchain-community                      0.4.1
# langchain-core                           1.0.3
# langchain-huggingface                    1.0.1
# langchain-ollama                         1.0.0
# langchain-openai                         1.0.2
# langchain-text-splitters                 1.0.0


# ------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Personal Assistant", layout="wide")
st.title("🤖 Thamim Personal Assistant")

uploaded_file = st.file_uploader("📂 Upload PDF (optional)", type=["pdf"])

# ------------------------------------------------------------------------------
# STEP 1: Initialize embeddings & Chroma
# ------------------------------------------------------------------------------
embeddings = HuggingFaceInstructEmbeddings(model_name=EMBED_MODEL)

if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    vectorstore = None

# ------------------------------------------------------------------------------
# STEP 2: If new file uploaded, process it
# ------------------------------------------------------------------------------
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    st.success("✅ PDF processed and knowledge base updated!")

# ------------------------------------------------------------------------------
# STEP 3: Setup Ollama model
# ------------------------------------------------------------------------------
llm = ChatOllama(model="mistral", temperature=0.3)

# ------------------------------------------------------------------------------
# STEP 4: Initialize chat history in session
# ------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------------------------------
# STEP 5: Retrieval and chat
# ------------------------------------------------------------------------------
if vectorstore is None:
    st.warning("⚠️ No PDF loaded yet. Upload one to start chatting.")
else:
    user_query = st.text_input("💬 Ask something ...")

    if user_query:
        # Build context from previous chat history
        context = ""
        for q, a in st.session_state["history"]:
            context += f"Q: {q}\nA: {a}\n"

        # Retrieve top documents from vectorstore
        docs = vectorstore.similarity_search(user_query, k=4)
        context_docs = "\n".join([doc.page_content[:500] for doc in docs])

        # Combine context + documents
        final_prompt = f"{context_docs}\n{context}\nQ: {user_query}\nA:"

        # Wrap prompt in HumanMessage
        prompt_message = HumanMessage(content=final_prompt)
        response = llm.generate([[prompt_message]])
        answer = response.generations[0][0].text

        # Save in session history
        st.session_state["history"].append((user_query, answer))

    # Display chat history
    for query, answer in st.session_state["history"]:
        st.markdown(f"**You:** {query}")
        st.markdown(f"**Your Assistant:** {answer}")
        st.markdown("---")

