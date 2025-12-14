"""
Streamlit UI for AI Notes Finder
A web interface for the RAG application
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from app import AINotesFinder

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Notes Finder",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = None
if 'processed' not in st.session_state:
    st.session_state.processed = False


def main():
    st.title("🔍 AI Notes Finder")
    st.markdown("**A Mini RAG Application** - Search and understand your notes using AI")
    
    st.sidebar.header("⚙️ Setup")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found. Please create a .env file with your API key.")
        st.sidebar.info("Create a `.env` file with:\n`OPENAI_API_KEY=your_key_here`")
        return
    
    st.sidebar.success("✅ API Key found")
    
    # Notes directory
    notes_dir = st.sidebar.text_input("Notes Directory", value="./notes")
    
    # Process button
    if st.sidebar.button("🔄 Process Notes", type="primary"):
        with st.spinner("Processing notes..."):
            try:
                if st.session_state.app is None:
                    st.session_state.app = AINotesFinder()
                
                st.session_state.app.process_notes(notes_dir, force_recreate=False)
                st.session_state.processed = True
                st.sidebar.success("✅ Notes processed!")
                st.success("🎉 Ready to answer questions!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.processed = False
    
    # Check if processed
    if not st.session_state.processed:
        st.info("👆 Click 'Process Notes' in the sidebar to get started")
        st.markdown("""
        ### How it works:
        1. Add your `.txt` files to the `notes` directory
        2. Click "Process Notes" to create embeddings
        3. Ask questions about your notes!
        """)
        return
    
    # Q&A Section
    st.header("💬 Ask Questions")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Process question
    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = st.session_state.app.ask(question)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.write(result['result'])
                
                # Display sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Information section
    with st.expander("ℹ️ About RAG"):
        st.markdown("""
        ### What is RAG?
        **Retrieval Augmented Generation** combines:
        - **Retrieval**: Finding relevant information from your notes
        - **Augmentation**: Adding context to the question
        - **Generation**: Using LLM to generate an answer
        
        ### How it works:
        1. Your notes are converted to **embeddings** (vector representations)
        2. Embeddings are stored in a **vector database** (ChromaDB)
        3. When you ask a question, similar chunks are **retrieved**
        4. The LLM uses these chunks as **context** to answer
        
        ### Key Concepts:
        - **Embeddings**: Text → Numbers that capture meaning
        - **Vector DB**: Fast similarity search
        - **LangChain**: Framework for LLM apps
        """)
    
    # Example questions
    st.sidebar.markdown("---")
    st.sidebar.header("💡 Example Questions")
    example_questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the key concepts in NLP?",
        "Summarize the main topics",
    ]
    
    for eq in example_questions:
        if st.sidebar.button(eq, key=f"example_{eq}"):
            st.session_state.question_input = eq
            st.rerun()


if __name__ == "__main__":
    main()

