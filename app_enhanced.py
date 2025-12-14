"""
Enhanced AI Notes Finder - Advanced RAG Application
Features:
- Multi-file type support (PDF, DOCX, Markdown, TXT)
- Conversation memory/history
- Streaming responses
- Multiple retrieval strategies
- Configuration file support
- Better error handling
- Query statistics
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    print("⚠️  Hybrid search not available. Install rank-bm25 for hybrid search.")

# Load environment variables
load_dotenv()


class EnhancedAINotesFinder:
    """Enhanced AI Notes Finder with advanced features"""
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retrieval_k: int = 3,
        use_hybrid_search: bool = False
    ):
        """
        Initialize the Enhanced AI Notes Finder
        
        Args:
            persist_directory: Directory to persist the vector database
            model: LLM model to use
            temperature: LLM temperature (0 = deterministic, higher = creative)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of chunks to retrieve
            use_hybrid_search: Use hybrid (semantic + keyword) search
        """
        self.persist_directory = persist_directory
        self.config = {
            "model": model,
            "temperature": temperature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_k": retrieval_k,
            "use_hybrid_search": use_hybrid_search
        }
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self.conversation_history = []
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "queries": []
        }
        
    def load_notes(self, notes_directory: str, file_types: Optional[List[str]] = None) -> List:
        """
        Load multiple file types from a directory
        
        Args:
            notes_directory: Path to directory containing files
            file_types: List of file extensions to load (default: ['txt', 'pdf', 'docx', 'md'])
            
        Returns:
            List of loaded documents
        """
        if file_types is None:
            file_types = ['txt', 'pdf', 'docx', 'md']
        
        print(f"📂 Loading notes from {notes_directory}...")
        
        if not os.path.exists(notes_directory):
            raise ValueError(f"Directory {notes_directory} does not exist")
        
        all_documents = []
        
        # Load different file types
        loaders = {
            'txt': (TextLoader, {}),
            'pdf': (PyPDFLoader, {}),
            'docx': (UnstructuredWordDocumentLoader, {}),
            'md': (UnstructuredMarkdownLoader, {})
        }
        
        for file_type in file_types:
            if file_type not in loaders:
                print(f"⚠️  Unsupported file type: {file_type}")
                continue
                
            loader_class, kwargs = loaders[file_type]
            try:
                loader = DirectoryLoader(
                    notes_directory,
                    glob=f"**/*.{file_type}",
                    loader_cls=loader_class,
                    show_progress=True,
                    loader_kwargs=kwargs
                )
                docs = loader.load()
                all_documents.extend(docs)
                print(f"✅ Loaded {len(docs)} {file_type.upper()} files")
            except Exception as e:
                print(f"⚠️  Error loading {file_type} files: {e}")
        
        print(f"✅ Total loaded: {len(all_documents)} documents")
        return all_documents
    
    def split_documents(
        self, 
        documents: List, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List:
        """
        Split documents into chunks with metadata preservation
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk (uses config if None)
            chunk_overlap: Overlap between chunks (uses config if None)
            
        Returns:
            List of document chunks
        """
        chunk_size = chunk_size or self.config['chunk_size']
        chunk_overlap = chunk_overlap or self.config['chunk_overlap']
        
        print(f"✂️ Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            if 'chunk_index' not in chunk.metadata:
                chunk.metadata['chunk_index'] = i
            if 'total_chunks' not in chunk.metadata:
                chunk.metadata['total_chunks'] = len(chunks)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List, force_recreate: bool = False):
        """
        Create or load vector store with optional hybrid search
        
        Args:
            chunks: List of document chunks
            force_recreate: If True, recreate the vector store
        """
        print(f"🔹 Creating vector store...")
        
        vectorstore_path = os.path.join(self.persist_directory, "faiss_index")
        index_file = os.path.join(vectorstore_path, "index.faiss")
        
        if os.path.exists(index_file) and not force_recreate:
            print(f"📦 Loading existing vector store from {self.persist_directory}...")
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"🆕 Creating new vector store...")
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            os.makedirs(vectorstore_path, exist_ok=True)
            self.vectorstore.save_local(vectorstore_path)
            print(f"✅ Vector store created and saved to {self.persist_directory}")
        
        # Store chunks for BM25 if hybrid search is enabled
        if self.config['use_hybrid_search']:
            self.all_chunks = chunks
    
    def setup_qa_chain(self, k: Optional[int] = None, use_streaming: bool = True):
        """
        Setup the QA chain with optional hybrid search and streaming
        
        Args:
            k: Number of relevant chunks to retrieve (uses config if None)
            use_streaming: Enable streaming responses
        """
        k = k or self.config['retrieval_k']
        print(f"🔗 Setting up QA chain (retrieval_k={k}, hybrid_search={self.config['use_hybrid_search']})...")
        
        # Create semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create hybrid retriever if enabled
        if self.config['use_hybrid_search'] and hasattr(self, 'all_chunks') and HYBRID_SEARCH_AVAILABLE:
            try:
                bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                bm25_retriever.k = k
                self.retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
                )
                print("✅ Using hybrid search (semantic + keyword)")
            except Exception as e:
                print(f"⚠️  Hybrid search failed, using semantic only: {e}")
                self.retriever = semantic_retriever
        else:
            if self.config['use_hybrid_search'] and not HYBRID_SEARCH_AVAILABLE:
                print("⚠️  Hybrid search requested but not available. Install rank-bm25.")
            self.retriever = semantic_retriever
        
        # Enhanced prompt with conversation history
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question. If you don't know the answer, 
just say that you don't know. Don't try to make up an answer.

Context:
{context}

Previous conversation:
{chat_history}

Answer in a clear and concise manner."""),
            ("human", "{question}")
        ])
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" 
                             for doc in docs)
        
        # Format chat history
        def format_history(messages):
            if not messages:
                return "No previous conversation."
            formatted = []
            for msg in messages[-5:]:  # Last 5 messages
                if isinstance(msg, HumanMessage):
                    formatted.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    formatted.append(f"Assistant: {msg.content}")
            return "\n".join(formatted)
        
        # Create QA chain
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "chat_history": lambda x: format_history(self.conversation_history),
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print(f"✅ QA chain ready!")
    
    def process_notes(
        self, 
        notes_directory: str, 
        force_recreate: bool = False,
        file_types: Optional[List[str]] = None
    ):
        """
        Complete pipeline: Load notes, split, create vectorstore, and setup QA chain
        
        Args:
            notes_directory: Path to directory containing files
            force_recreate: If True, recreate the vector store
            file_types: List of file extensions to load
        """
        print("\n" + "="*50)
        print("🚀 Starting Enhanced AI Notes Finder Pipeline")
        print("="*50 + "\n")
        
        # Step 1: Load notes
        documents = self.load_notes(notes_directory, file_types=file_types)
        
        if not documents:
            print("⚠️  No documents found!")
            return
        
        # Step 2: Split documents
        chunks = self.split_documents(documents)
        
        # Step 3: Create vector store
        self.create_vectorstore(chunks, force_recreate=force_recreate)
        
        # Step 4: Setup QA chain
        self.setup_qa_chain()
        
        print("\n" + "="*50)
        print("✅ Pipeline Complete! Ready to answer questions.")
        print("="*50 + "\n")
    
    def ask(
        self, 
        question: str, 
        stream: bool = False,
        add_to_history: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with statistics
        
        Args:
            question: The question to ask
            stream: If True, return streaming response
            add_to_history: If True, add to conversation history
            
        Returns:
            Dictionary with answer, source documents, and stats
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Run process_notes() first.")
        
        self.query_stats["total_queries"] += 1
        start_time = datetime.now()
        
        try:
            print(f"\n❓ Question: {question}")
            print("🔍 Searching for relevant information...\n")
            
            # Get answer from chain
            if stream:
                answer_chunks = []
                for chunk in self.qa_chain.stream(question):
                    answer_chunks.append(chunk)
                    print(chunk, end='', flush=True)
                answer = "".join(answer_chunks)
                print("\n")
            else:
                answer = self.qa_chain.invoke(question)
            
            # Get source documents
            source_documents = self.retriever.invoke(question)
            
            # Calculate time
            end_time = datetime.now()
            query_time = (end_time - start_time).total_seconds()
            
            # Update stats
            self.query_stats["successful_queries"] += 1
            self.query_stats["queries"].append({
                "question": question,
                "timestamp": start_time.isoformat(),
                "time_taken": query_time,
                "sources_count": len(source_documents),
                "status": "success"
            })
            
            # Add to conversation history
            if add_to_history:
                self.conversation_history.append(HumanMessage(content=question))
                self.conversation_history.append(AIMessage(content=answer))
            
            print(f"💡 Answer: {answer}\n")
            print(f"📚 Sources ({len(source_documents)} chunks):")
            for i, doc in enumerate(source_documents, 1):
                source = doc.metadata.get('source', 'Unknown')
                if os.path.exists(source):
                    source = os.path.basename(source)
                print(f"  {i}. {source}")
            
            print(f"⏱️  Query time: {query_time:.2f}s\n")
            
            return {
                "result": answer,
                "source_documents": source_documents,
                "query_time": query_time,
                "sources_count": len(source_documents),
                "status": "success"
            }
            
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            self.query_stats["queries"].append({
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ Error: {e}\n")
            raise
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics"""
        return {
            **self.query_stats,
            "conversation_length": len(self.conversation_history),
            "success_rate": (
                self.query_stats["successful_queries"] / self.query_stats["total_queries"]
                if self.query_stats["total_queries"] > 0 else 0
            )
        }
    
    def save_config(self, config_path: str = "config.json"):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to {config_path}")
    
    def load_config(self, config_path: str = "config.json"):
        """Load configuration from file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
            print(f"✅ Configuration loaded from {config_path}")
            # Reinitialize with new config
            self.llm = ChatOpenAI(
                model=self.config['model'], 
                temperature=self.config['temperature'],
                streaming=True
            )
            return True
        return False


def main():
    """Main function with enhanced features"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize the enhanced app
    app = EnhancedAINotesFinder(
        model="gpt-3.5-turbo",
        temperature=0,
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=3,
        use_hybrid_search=False  # Set to True for hybrid search
    )
    
    # Try to load config
    app.load_config()
    
    # Process notes
    notes_dir = "./notes"
    
    if not os.path.exists(notes_dir):
        print(f"📁 Creating {notes_dir} directory...")
        os.makedirs(notes_dir, exist_ok=True)
        print(f"✅ Please add your files to {notes_dir} and run again.")
        return
    
    # Process notes (supports txt, pdf, docx, md)
    app.process_notes(notes_dir, file_types=['txt', 'pdf', 'docx', 'md'])
    
    # Interactive Q&A loop
    print("\n" + "="*50)
    print("💬 Enhanced Interactive Q&A Mode")
    print("Commands:")
    print("  - Type your question to ask")
    print("  - 'stats' - Show query statistics")
    print("  - 'clear' - Clear conversation history")
    print("  - 'quit' or 'exit' - Stop")
    print("="*50 + "\n")
    
    while True:
        question = input("Ask a question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            stats = app.get_stats()
            print(f"\n📊 Final Stats:")
            print(f"  Total queries: {stats['total_queries']}")
            print(f"  Successful: {stats['successful_queries']}")
            print(f"  Failed: {stats['failed_queries']}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            break
        
        if question.lower() == 'stats':
            stats = app.get_stats()
            print(f"\n📊 Query Statistics:")
            print(f"  Total queries: {stats['total_queries']}")
            print(f"  Successful: {stats['successful_queries']}")
            print(f"  Failed: {stats['failed_queries']}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"  Conversation length: {stats['conversation_length']} messages\n")
            continue
        
        if question.lower() == 'clear':
            app.clear_history()
            continue
        
        if not question:
            continue
        
        try:
            app.ask(question, stream=False)
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()

