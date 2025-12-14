"""
Enhanced Streamlit UI for AI Notes Finder
A beautiful web interface with all enhanced features
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

from app_enhanced import EnhancedAINotesFinder

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Notes Finder - Enhanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'config' not in st.session_state:
    st.session_state.config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval_k": 3,
        "use_hybrid_search": False
    }


def display_conversation():
    """Display conversation history"""
    if st.session_state.conversation:
        st.markdown("### 💬 Conversation History")
        for i, msg in enumerate(st.session_state.conversation):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if "sources" in msg:
                        with st.expander("📚 View Sources"):
                            for j, source in enumerate(msg["sources"], 1):
                                st.markdown(f"**{j}. {source}**")


def main():
    # Header
    st.markdown('<div class="main-header">🔍 AI Notes Finder - Enhanced</div>', unsafe_allow_html=True)
    st.markdown("**Advanced RAG Application** - Search and understand your notes using AI")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found. Please create a .env file with your API key.")
            st.info("Create a `.env` file with:\n`OPENAI_API_KEY=your_key_here`")
            return
        
        st.success("✅ API Key found")
        st.divider()
        
        # Model selection
        st.subheader("🤖 Model Settings")
        model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, 
                               help="Higher = more creative, Lower = more deterministic")
        
        st.divider()
        
        # Chunking settings
        st.subheader("📄 Chunking Settings")
        chunk_size = st.slider("Chunk Size", 500, 3000, 1000, 100,
                              help="Size of document chunks in characters")
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50,
                                 help="Overlap between chunks")
        
        st.divider()
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        retrieval_k = st.slider("Number of Chunks (k)", 1, 10, 3,
                               help="How many chunks to retrieve")
        use_hybrid_search = st.checkbox(
            "Enable Hybrid Search",
            value=False,
            help="Combine semantic + keyword search (requires rank-bm25)"
        )
        
        st.divider()
        
        # Notes directory
        st.subheader("📁 Notes Directory")
        notes_dir = st.text_input("Path to Notes", value="./notes")
        
        # File type selection
        st.subheader("📄 File Types")
        file_types = st.multiselect(
            "Select file types to process",
            ["txt", "pdf", "docx", "md"],
            default=["txt"],
            help="Select which file types to include"
        )
        
        # Update config
        st.session_state.config = {
            "model": model,
            "temperature": temperature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_k": retrieval_k,
            "use_hybrid_search": use_hybrid_search
        }
        
        st.divider()
        
        # Process button
        if st.button("🔄 Process Notes", type="primary", use_container_width=True):
            with st.spinner("Processing notes..."):
                try:
                    # Initialize app with config
                    st.session_state.app = EnhancedAINotesFinder(
                        model=st.session_state.config["model"],
                        temperature=st.session_state.config["temperature"],
                        chunk_size=st.session_state.config["chunk_size"],
                        chunk_overlap=st.session_state.config["chunk_overlap"],
                        retrieval_k=st.session_state.config["retrieval_k"],
                        use_hybrid_search=st.session_state.config["use_hybrid_search"]
                    )
                    
                    if not file_types:
                        st.warning("⚠️ Please select at least one file type")
                        st.stop()
                    
                    st.session_state.app.process_notes(
                        notes_dir, 
                        file_types=file_types,
                        force_recreate=False
                    )
                    st.session_state.processed = True
                    st.success("✅ Notes processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.processed = False
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation = []
            if st.session_state.app:
                st.session_state.app.clear_history()
            st.success("✅ Conversation cleared!")
            st.rerun()
        
        st.divider()
        
        # Statistics
        if st.session_state.app and st.session_state.processed:
            st.subheader("📊 Statistics")
            try:
                stats = st.session_state.app.get_stats()
                st.metric("Total Queries", stats.get("total_queries", 0))
                st.metric("Success Rate", f"{stats.get('success_rate', 0)*100:.1f}%")
                st.metric("Conversation Length", stats.get("conversation_length", 0))
                
                if stats.get("total_queries", 0) > 0:
                    with st.expander("📈 View Detailed Stats"):
                        st.json(stats)
            except:
                pass
    
    # Main content area
    if not st.session_state.processed:
        st.info("👆 Please configure settings and click 'Process Notes' in the sidebar to get started")
        
        # Show configuration info
        with st.expander("ℹ️ About Enhanced Features"):
            st.markdown("""
            ### 🚀 Enhanced Features
            
            **1. Multi-File Support**
            - Supports TXT, PDF, DOCX, and Markdown files
            - Process multiple document types simultaneously
            
            **2. Conversation Memory**
            - Remembers previous questions and answers
            - Enables contextual follow-up questions
            
            **3. Hybrid Search** (Optional)
            - Combines semantic + keyword search
            - Better retrieval accuracy
            
            **4. Streaming Responses**
            - See answers as they generate
            - Better user experience
            
            **5. Query Statistics**
            - Track all queries
            - Monitor success rates
            - View performance metrics
            
            **6. Advanced Configuration**
            - Adjust model settings
            - Tune chunking parameters
            - Customize retrieval settings
            """)
        
        return
    
    # Main chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Display conversation history
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(msg["sources"], 1):
                            st.markdown(f"**{i}.** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your notes..."):
        # Add user message to conversation
        st.session_state.conversation.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ask question
                    result = st.session_state.app.ask(prompt, stream=False, add_to_history=True)
                    
                    # Display answer
                    answer = result.get("result", "")
                    st.write(answer)
                    
                    # Display sources
                    sources = result.get("source_documents", [])
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, doc in enumerate(sources, 1):
                                source = doc.metadata.get('source', 'Unknown')
                                if os.path.exists(source):
                                    source = os.path.basename(source)
                                st.markdown(f"**{i}.** {source}")
                                # Show snippet
                                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.caption(content_preview)
                    
                    # Display query time
                    query_time = result.get("query_time", 0)
                    st.caption(f"⏱️ Query took {query_time:.2f} seconds")
                    
                    # Add to conversation
                    source_names = [doc.metadata.get('source', 'Unknown') for doc in sources]
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_names,
                        "query_time": query_time
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Example questions
    st.divider()
    st.markdown("### 💡 Example Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    example_questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the key concepts in NLP?",
        "Summarize the main topics",
    ]
    
    for i, question in enumerate(example_questions):
        with [col1, col2, col3, col4][i]:
            if st.button(question, use_container_width=True, key=f"example_{i}"):
                # Simulate chat input
                st.session_state.chat_input = question
                st.rerun()
    
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
        2. Embeddings are stored in a **vector database** (FAISS)
        3. When you ask a question, similar chunks are **retrieved**
        4. The LLM uses these chunks as **context** to answer
        
        ### Key Concepts:
        - **Embeddings**: Text → Numbers that capture meaning
        - **Vector DB**: Fast similarity search
        - **LangChain**: Framework for LLM apps
        - **Hybrid Search**: Semantic + Keyword search combined
        """)


if __name__ == "__main__":
    main()

