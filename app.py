"""
AI Notes Finder - Mini RAG Application
Converts text files to embeddings, stores in vector DB, and answers questions using RAG
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


class AINotesFinder:
    """Main class for the AI Notes Finder RAG application"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the AI Notes Finder
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        self.qa_chain = None
        
    def load_notes(self, notes_directory: str) -> List:
        """
        Load text files from a directory
        
        Args:
            notes_directory: Path to directory containing text files
            
        Returns:
            List of loaded documents
        """
        print(f"📂 Loading notes from {notes_directory}...")
        
        if not os.path.exists(notes_directory):
            raise ValueError(f"Directory {notes_directory} does not exist")
        
        # Load all text files from directory
        loader = DirectoryLoader(
            notes_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        print(f"✂️ Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List, force_recreate: bool = False):
        """
        Create or load vector store from document chunks
        
        Args:
            chunks: List of document chunks
            force_recreate: If True, recreate the vector store even if it exists
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
            # Create directory if it doesn't exist
            os.makedirs(vectorstore_path, exist_ok=True)
            self.vectorstore.save_local(vectorstore_path)
            print(f"✅ Vector store created and saved to {self.persist_directory}")
    
    def setup_qa_chain(self, k: int = 3):
        """
        Setup the QA chain for question answering using LCEL (LangChain Expression Language)
        
        Args:
            k: Number of relevant chunks to retrieve
        """
        print(f"🔗 Setting up QA chain...")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Custom prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create QA chain using LCEL
        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Store retriever for source documents
        self.retriever = retriever
        
        print(f"✅ QA chain ready!")
    
    def process_notes(self, notes_directory: str, force_recreate: bool = False):
        """
        Complete pipeline: Load notes, split, create vectorstore, and setup QA chain
        
        Args:
            notes_directory: Path to directory containing text files
            force_recreate: If True, recreate the vector store
        """
        print("\n" + "="*50)
        print("🚀 Starting AI Notes Finder Pipeline")
        print("="*50 + "\n")
        
        # Step 1: Load notes
        documents = self.load_notes(notes_directory)
        
        # Step 2: Split documents
        chunks = self.split_documents(documents)
        
        # Step 3: Create vector store
        self.create_vectorstore(chunks, force_recreate=force_recreate)
        
        # Step 4: Setup QA chain
        self.setup_qa_chain()
        
        print("\n" + "="*50)
        print("✅ Pipeline Complete! Ready to answer questions.")
        print("="*50 + "\n")
    
    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Run process_notes() first.")
        
        print(f"\n❓ Question: {question}")
        print("🔍 Searching for relevant information...\n")
        
        # Get answer from chain
        answer = self.qa_chain.invoke(question)
        
        # Get source documents (retriever is callable in newer versions)
        source_documents = self.retriever.invoke(question)
        
        print(f"💡 Answer: {answer}\n")
        print(f"📚 Sources ({len(source_documents)} chunks):")
        for i, doc in enumerate(source_documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            # Clean up source path for display
            if os.path.exists(source):
                source = os.path.basename(source)
            print(f"  {i}. {source}")
        
        return {
            "result": answer,
            "source_documents": source_documents
        }


def main():
    """Main function to run the application"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize the app
    app = AINotesFinder()
    
    # Process notes (assuming notes are in ./notes directory)
    notes_dir = "./notes"
    
    if not os.path.exists(notes_dir):
        print(f"📁 Creating {notes_dir} directory...")
        os.makedirs(notes_dir, exist_ok=True)
        print(f"✅ Please add your .txt files to {notes_dir} and run again.")
        return
    
    # Process notes
    app.process_notes(notes_dir)
    
    # Interactive Q&A loop
    print("\n" + "="*50)
    print("💬 Interactive Q&A Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        question = input("Ask a question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            app.ask(question)
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()

