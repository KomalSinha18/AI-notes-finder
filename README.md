# AI Notes Finder - Mini RAG Application

A simple Retrieval Augmented Generation (RAG) application that helps you search and understand your notes using AI.

## 🎯 What This App Does

1. **Loads** text files (your notes)
2. **Converts** text → embeddings (vector representations)
3. **Stores** embeddings in a vector database (ChromaDB)
4. **Retrieves** relevant text chunks when you ask questions
5. **Answers** using an LLM (OpenAI GPT) with context from your notes

## 🧠 Concepts You'll Learn

- **Embeddings**: Converting text into numerical vectors that capture meaning
- **Vector Database**: Storing and searching through embeddings efficiently
- **LangChain**: Framework for building LLM applications
- **RAG (Retrieval Augmented Generation)**: Combining retrieval with LLM generation

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key

## 🚀 Setup

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Add your notes**:
   
   Create a `notes` folder and add your `.txt` files:
   ```
   notes/
     ├── note1.txt
     ├── note2.txt
     └── ...
   ```

## 💻 Usage

### Command Line Interface

Run the main application:
```bash
python app.py
```

The app will:
1. Load all `.txt` files from the `notes` directory
2. Process them into embeddings and store in ChromaDB
3. Start an interactive Q&A session

### Example Usage

```
Ask a question: What is machine learning?
Ask a question: Summarize the key points about neural networks
Ask a question: quit
```

### Programmatic Usage

```python
from app import AINotesFinder

# Initialize
app = AINotesFinder()

# Process notes
app.process_notes("./notes")

# Ask questions
result = app.ask("What is the main topic?")
print(result['result'])
```

## 📁 Project Structure

```
ai-notes-finder/
├── app.py              # Main application code
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .env               # Your API keys (create this)
├── notes/             # Your text files go here
└── chroma_db/         # Vector database (auto-created)
```

## 🔧 How It Works

1. **Document Loading**: Uses LangChain's `DirectoryLoader` to load all `.txt` files
2. **Text Splitting**: Splits documents into chunks (1000 chars, 200 overlap)
3. **Embedding**: Converts chunks to embeddings using OpenAI's embedding model
4. **Vector Storage**: Stores embeddings in ChromaDB (persistent storage)
5. **Retrieval**: When you ask a question, finds the most relevant chunks
6. **Generation**: Uses GPT-3.5 to generate an answer based on retrieved context

## 🎓 Learning Points

### Embeddings
- Text is converted to high-dimensional vectors
- Similar texts have similar vectors
- Enables semantic search (not just keyword matching)

### Vector Database
- ChromaDB stores embeddings efficiently
- Enables fast similarity search
- Persists data to disk for reuse

### RAG Flow
```
User Question → Embed Question → Search Vector DB → Retrieve Relevant Chunks → 
LLM with Context → Generate Answer
```

## 🛠️ Customization

### Change Chunk Size
```python
chunks = app.split_documents(documents, chunk_size=500, chunk_overlap=100)
```

### Change Number of Retrieved Chunks
```python
app.setup_qa_chain(k=5)  # Retrieve top 5 chunks
```

### Use Different LLM
Modify the `__init__` method in `app.py`:
```python
self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
```

## 📝 Example Notes

Check the `notes/` directory for example text files. You can add your own notes there!

## ⚠️ Notes

- First run will create the vector database
- Subsequent runs will reuse the existing database (unless you delete `chroma_db/`)
- API costs: You'll be charged for OpenAI API usage (embeddings + LLM calls)

## 🐛 Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure you created a `.env` file with your API key

**"Directory does not exist"**
- Create a `notes` folder and add some `.txt` files

**Import errors**
- Make sure you installed all dependencies: `pip install -r requirements.txt`

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 🎉 Enjoy!

Start adding your notes and asking questions!

